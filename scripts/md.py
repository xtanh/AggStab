#!/usr/bin/env python
# md_simulation.py - 分子动力学模拟脚本（自动识别二硫键）

from openmm import *
from openmm.app import *
import openmm.unit as u
import builtins
from pdbfixer import PDBFixer
import sys
import time
import argparse
import os
import random
import numpy as np
from math import sqrt

# 蛋白质标准三字母表
PROT_3 = {
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY',
    'HIS','ILE','LEU','LYS','MET','PHE','PRO','SER',
    'THR','TRP','TYR','VAL','CYX'
}

def run_simulation(input_pdb, output_prefix, simulation_time_ns=100, seed=None, disulfide_cutoff_nm=0.25):
    print(f"开始处理: {input_pdb}")
    print(f"输出前缀: {output_prefix}")
    print(f"模拟时间: {simulation_time_ns} ns")
    print(f"二硫键识别阈值: {disulfide_cutoff_nm:.3f} nm")

    # 随机种子
    if seed is not None:
        print(f"使用随机种子: {seed}")
        random.seed(seed)
        np.random.seed(seed)

    # 平台
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu_id is not None:
        print(f"使用GPU: {gpu_id}")
    platforms = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    print(f"可用的OpenMM平台: {platforms}")
    if 'CUDA' in platforms:
        platform = Platform.getPlatformByName('CUDA')
        properties = {'Precision': 'mixed'}
        print("使用CUDA平台")
    elif 'OpenCL' in platforms:
        platform = Platform.getPlatformByName('OpenCL')
        properties = {'Precision': 'mixed'}
        print("使用OpenCL平台")
    else:
        platform = Platform.getPlatformByName('CPU')
        properties = {}
        print("使用CPU平台")

    # ---------- 结构修复（不加氢） ----------
    print("修复PDB/CIF文件（补残基/原子，不加氢）...")
    fixer = PDBFixer(input_pdb)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    fixed_pdb = f"{output_prefix}_fixed.pdb"
    with open(fixed_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    print(f"已保存修复后的PDB文件: {fixed_pdb}")

    pdb = PDBFile(fixed_pdb)

    # 链/残基统计
    print("修复后PDB文件中的链:")
    chain_residues = {}
    for chain in pdb.topology.chains():
        cid = chain.id
        chain_residues[cid] = {"protein": 0, "other": 0}
        for res in chain.residues():
            if res.name in PROT_3:
                chain_residues[cid]["protein"] += 1
            else:
                chain_residues[cid]["other"] += 1
    for cid, c in chain_residues.items():
        print(f"链 {cid}: {c['protein']} 个蛋白质残基, {c['other']} 个其他残基")

    modeller = Modeller(pdb.topology, pdb.positions)

    protein_chains = {cid: c["protein"] for cid, c in chain_residues.items() if c["protein"] > 0}
    if not protein_chains:
        print("错误：未找到包含蛋白质的链!")
        sys.exit(1)
    keep = [max(protein_chains, key=protein_chains.get)]
    print(f"保留链: {keep}")
    to_delete = [chain for chain in modeller.topology.chains() if chain.id not in keep]
    if to_delete:
        modeller.delete(to_delete)
        print(f"删除了 {len(to_delete)} 个非目标链")

    print("检查残基类型:")
    residue_types = {}
    for res in modeller.topology.residues():
        residue_types[res.name] = residue_types.get(res.name, 0) + 1
    print(residue_types)

    protein_residues = builtins.sum(1 for res in modeller.topology.residues() if res.name in PROT_3)
    if protein_residues == 0:
        print("警告：系统中没有蛋白质残基！")
        sys.exit(1)
    else:
        print(f"系统中包含 {protein_residues} 个蛋白质残基，继续模拟...")

    # ---------- 识别二硫键 ----------
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')

    print("按距离搜索 CYS–CYS 以识别二硫键...")
    positions = modeller.positions
    cys_list = []  # [(Residue, Atom)]
    for res in modeller.topology.residues():
        if res.name == 'CYS':
            for atom in res.atoms():
                if atom.name == 'SG':
                    cys_list.append((res, atom))
                    break

    # 识别配对
    cys_residue_pairs = []
    used_indices = set()
    for i in range(len(cys_list)):
        if i in used_indices:
            continue
        ri, ai = cys_list[i]
        pi = positions[ai.index].value_in_unit(u.nanometer)
        for j in range(i + 1, len(cys_list)):
            if j in used_indices:
                continue
            rj, aj = cys_list[j]
            pj = positions[aj.index].value_in_unit(u.nanometer)
            d = sqrt((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2 + (pi[2]-pj[2])**2)
            if d < disulfide_cutoff_nm:
                cys_residue_pairs.append((ri, rj, d))
                used_indices.add(i)
                used_indices.add(j)
                break

    if cys_residue_pairs:
        print(f"识别到 {len(cys_residue_pairs)} 对潜在的二硫键:")
        for r1, r2, d in cys_residue_pairs:
            print(f"  链{r1.chain.id}:{r1.id}.SG <-> 链{r2.chain.id}:{r2.id}.SG，距离 {d*10:.2f} Å")
    else:
        print("⚠️ 未发现满足阈值的 CYS–CYS 对；所有 CYS 将保持还原态。")

    # ###############################################################
    # ##########          核心逻辑修正部分 START          ############
    # ###############################################################

    # 1. 首先对整个系统加氢。此时，所有半胱氨酸都被视为标准的 CYS，并会被加上 HG 氢原子。
    print("添加氢原子 (pH=7.0)...")
    modeller.addHydrogens(forcefield, pH=7.0)
    print("已完成加氢。")

    # 2. 如果之前识别到了二硫键对，现在对这些特定的残基进行处理。
    if cys_residue_pairs:
        print("处理二硫键：删除 HG 并改名为 CYX...")
        
        # 收集所有需要修改的残基
        residues_to_modify = set()
        for r1, r2, d in cys_residue_pairs:
            residues_to_modify.add(r1)
            residues_to_modify.add(r2)

        atoms_to_delete = []
        # 遍历拓扑结构，修改残基名并记录要删除的HG原子
        for res in modeller.topology.residues():
            if res in residues_to_modify:
                res.name = 'CYX'
                for atom in res.atoms():
                    if atom.name == 'HG':
                        atoms_to_delete.append(atom)
                        break
        
        # 执行删除操作
        if atoms_to_delete:
            modeller.delete(atoms_to_delete)
            print(f"已删除 {len(atoms_to_delete)} 个 HG 氢原子。")
        
        print(f"已将 {len(residues_to_modify)} 个 CYS 残基改名为 CYX。")

    # ###############################################################
    # ##########           核心逻辑修正部分 END           ############
    # ###############################################################


    # ---------- 溶剂化与建系统 ----------
    print("添加溶剂盒子...")
    try:
        modeller.addSolvent(forcefield,
                            model='tip3p',
                            padding=1.0*u.nanometer,
                            ionicStrength=0.15*u.molar)
        print("成功添加溶剂盒子")
    except Exception as e:
        print(f"添加溶剂盒子时出错: {e}")
        sys.exit(1)

    print("创建溶剂系统...")
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=PME,
                                     nonbondedCutoff=1.0*u.nanometer,
                                     constraints=HBonds) # 删除了 ignoreExternalBonds=True，通常不需要

    temperature = 310.15*u.kelvin
    print(f"设置模拟温度为 {temperature.value_in_unit(u.kelvin):.2f} K ({temperature.value_in_unit(u.kelvin)-273.15:.2f} °C)")
    integrator = LangevinIntegrator(temperature, 1/u.picosecond, 0.002*u.picoseconds)
    if seed is not None:
        integrator.setRandomNumberSeed(seed)

    if platform.getName() in ['CUDA', 'OpenCL']:
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    else:
        simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    print(f"使用平台: {simulation.context.getPlatform().getName()}")
    if simulation.context.getPlatform().getName() in ['CUDA', 'OpenCL']:
        try:
            if simulation.context.getPlatform().getName() == 'CUDA':
                device_index = simulation.context.getPlatform().getPropertyValue(simulation.context, 'DeviceIndex')
                print(f"设备ID: {device_index}")
            precision = simulation.context.getPlatform().getPropertyValue(simulation.context, 'Precision')
            print(f"精度模式: {precision}")
        except Exception as e:
            print(f"无法获取平台详细属性: {e}")

    # 组成统计
    total_atoms = protein_atoms = water_atoms = ion_atoms = 0
    for atom in modeller.topology.atoms():
        total_atoms += 1
        r = atom.residue
        if r.name in PROT_3:
            protein_atoms += 1
        elif r.name in ('HOH', 'WAT'):
            water_atoms += 1
        elif r.name in ('NA', 'CL', 'K', 'CA', 'MG'):
            ion_atoms += 1
    print(f"系统组成: 总原子数 = {total_atoms}, 蛋白质原子 = {protein_atoms}, 水分子原子 = {water_atoms}, 离子原子 = {ion_atoms}")

    # 最小化
    print("进行能量最小化...")
    simulation.minimizeEnergy()
    print("能量最小化完成")
    positions_state = simulation.context.getState(getPositions=True).getPositions()
    minimized_pdb = f"{output_prefix}_minimized.pdb"
    with open(minimized_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, positions_state, f)
    print(f"已保存最小化后的结构: {minimized_pdb}")

    # NVT
    print("进行NVT平衡...")
    simulation.context.setVelocitiesToTemperature(temperature)
    t0 = time.time()
    simulation.step(50000)
    nvt_time = time.time() - t0
    print(f"NVT平衡完成，用时: {nvt_time:.2f}秒")
    positions_state = simulation.context.getState(getPositions=True).getPositions()
    nvt_pdb = f"{output_prefix}_nvt.pdb"
    with open(nvt_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, positions_state, f)
    print(f"已保存NVT平衡后的结构: {nvt_pdb}")

    # NPT
    print("进行NPT平衡...")
    system_npt = forcefield.createSystem(modeller.topology,
                                        nonbondedMethod=PME,
                                        nonbondedCutoff=1.0*u.nanometer,
                                        constraints=HBonds)
    system_npt.addForce(MonteCarloBarostat(1.0*u.bar, temperature))
    integrator_npt = LangevinIntegrator(temperature, 1/u.picosecond, 0.002*u.picoseconds)
    if seed is not None:
        integrator_npt.setRandomNumberSeed(seed)
    if platform.getName() in ['CUDA', 'OpenCL']:
        simulation_npt = Simulation(modeller.topology, system_npt, integrator_npt, platform, properties)
    else:
        simulation_npt = Simulation(modeller.topology, system_npt, integrator_npt, platform)
    simulation_npt.context.setPositions(positions_state)
    simulation_npt.context.setVelocitiesToTemperature(temperature)
    t0 = time.time()
    simulation_npt.step(50000)
    npt_time = time.time() - t0
    print(f"NPT平衡完成，用时: {npt_time:.2f}秒")
    positions_state = simulation_npt.context.getState(getPositions=True).getPositions()
    equilibrated_pdb = f"{output_prefix}_equilibrated.pdb"
    with open(equilibrated_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, positions_state, f)
    print(f"已保存平衡后的结构: {equilibrated_pdb}")

    # 生产阶段继续使用 NPT 平衡后的 simulation 对象，保证状态连续
    production_simulation = simulation_npt

    # 报告器
    dcd_file = f"{output_prefix}_trajectory.dcd"
    log_file = f"{output_prefix}_output.txt"
    frames_file = f"{output_prefix}_frames.pdb"
    production_simulation.reporters.append(DCDReporter(dcd_file, 5000))
    production_simulation.reporters.append(StateDataReporter(log_file, 5000, step=True,
        potentialEnergy=True, temperature=True, totalEnergy=True, volume=True, speed=True))
    production_simulation.reporters.append(PDBReporter(frames_file, 25000))
    checkpoint_file = f"{output_prefix}_checkpoint.chk"
    production_simulation.reporters.append(CheckpointReporter(checkpoint_file, 100000))

    # 生产
    steps_per_ns = 500000
    total_steps = simulation_time_ns * steps_per_ns
    print(f"开始生产模拟 ({simulation_time_ns} ns)...")
    t_all = time.time()
    for stage in range(10):
        stage_steps = total_steps // 10
        t0 = time.time()
        production_simulation.step(stage_steps)
        dt = time.time() - t0
        ns_per_day = (stage_steps * 0.002e-3) / dt * 86400.0
        print(f"完成阶段 {stage+1}/10 ({(stage+1)*10}%)，用时: {dt:.2f}s，性能: {ns_per_day:.2f} ns/day")

    sim_time = time.time() - t_all
    ns_per_day = (total_steps * 0.002e-3) / sim_time * 86400.0
    print(f"生产模拟完成，总用时: {sim_time:.2f}秒")
    print(f"模拟性能: {ns_per_day:.2f} ns/day")

    # 最终结构
    positions_state = production_simulation.context.getState(getPositions=True).getPositions()
    final_pdb = f"{output_prefix}_final.pdb"
    with open(final_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, positions_state, f)
    print(f"溶剂模拟完成，已保存最终结构: {final_pdb}")

    # 汇总
    print("\n模拟统计信息:")
    print(f"总模拟长度: {simulation_time_ns} 纳秒 ({total_steps}步)")
    print(f"温度: {(temperature.value_in_unit(u.kelvin) - 273.15):.2f} °C ({temperature.value_in_unit(u.kelvin):.2f} K)")
    print(f"总原子数: {total_atoms}")
    print(f"蛋白质原子数: {protein_atoms}")
    print(f"水和离子原子数: {water_atoms + ion_atoms}")
    print(f"NVT平衡时间: {nvt_time:.2f}秒")
    print(f"NPT平衡时间: {npt_time:.2f}秒")
    print(f"生产模拟时间: {sim_time:.2f}秒")
    print(f"模拟性能: {ns_per_day:.2f} ns/day")
    print("输出文件:")
    print(f"  轨迹: {dcd_file}")
    print(f"  日志: {log_file}")
    print(f"  结构帧: {frames_file}")
    print(f"  最终结构: {final_pdb}")

    return final_pdb, dcd_file, equilibrated_pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行蛋白质分子动力学模拟（自动识别二硫键）')
    parser.add_argument('--input', required=True, help='输入PDB/CIF文件路径')
    parser.add_argument('--output', required=True, help='输出文件前缀')
    parser.add_argument('--time', type=int, default=100, help='模拟时间(纳秒)')
    parser.add_argument('--seed', type=int, help='随机数种子(可选)')
    parser.add_argument('--ss-cutoff', type=float, default=0.25, help='二硫键SG–SG距离阈值(单位nm)')
    args = parser.parse_args()

    run_simulation(args.input, args.output, args.time, args.seed, args.ss_cutoff)
