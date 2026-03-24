# @sylphx/vex-json-schema

JSON Schema conversion for [@sylphx/vex](https://github.com/SylphxAI/vex) validators.

## Installation

```bash
npm install @sylphx/vex-json-schema
# or
bun add @sylphx/vex-json-schema
```

## Usage

```typescript
import { str, num, object, optional, pipe } from '@sylphx/vex'
import { email, int, gte } from '@sylphx/vex'
import { toJsonSchema } from '@sylphx/vex-json-schema'

// Simple schema
const emailSchema = pipe(str(), email)
toJsonSchema(emailSchema)
// { "$schema": "http://json-schema.org/draft-07/schema#", "type": "string", "format": "email" }

// Object schema
const userSchema = object({
  name: str(),
  email: pipe(str(), email),
  age: optional(pipe(num(), int, gte(0))),
})

toJsonSchema(userSchema)
// {
//   "$schema": "http://json-schema.org/draft-07/schema#",
//   "type": "object",
//   "properties": {
//     "name": { "type": "string" },
//     "email": { "type": "string", "format": "email" },
//     "age": { "type": "integer", "minimum": 0 }
//   },
//   "required": ["name", "email"]
// }
```

## API

### `toJsonSchema(schema, options?)`

Convert a Vex schema to JSON Schema.

```typescript
toJsonSchema(schema: Schema<unknown>, options?: ToJsonSchemaOptions): JsonSchema
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `$schema` | `boolean` | `true` | Include `$schema` property |
| `draft` | `'draft-07' \| 'draft-2019-09' \| 'draft-2020-12'` | `'draft-07'` | JSON Schema draft version |
| `definitions` | `Record<string, Schema>` | - | Named definitions to include as `$defs` |

### `toJsonSchemaDefs(definitions)`

Convert multiple schemas to JSON Schema definitions without a root schema.

```typescript
toJsonSchemaDefs(definitions: Record<string, Schema<unknown>>): Record<string, JsonSchema>
```

### Global Definitions

```typescript
import { addGlobalDefs, getGlobalDefs, clearGlobalDefs } from '@sylphx/vex-json-schema'

// Add global definitions
addGlobalDefs({
  User: userSchema,
  Post: postSchema,
})

// Get current definitions
const defs = getGlobalDefs()

// Clear all definitions
clearGlobalDefs()
```

## Supported Types

| Vex Type | JSON Schema |
|----------|-------------|
| `str()` | `{ type: "string" }` |
| `num()` | `{ type: "number" }` |
| `bool()` | `{ type: "boolean" }` |
| `nullType()` | `{ type: "null" }` |
| `bigInt()` | `{ type: "integer" }` |
| `date()` | `{ type: "string", format: "date-time" }` |
| `array(T)` | `{ type: "array", items: T }` |
| `object({...})` | `{ type: "object", properties: {...} }` |
| `tuple(...)` | `{ type: "array", items: [...], minItems, maxItems }` |
| `record(K, V)` | `{ type: "object", additionalProperties: V }` |
| `set(T)` | `{ type: "array", uniqueItems: true, items: T }` |
| `union(...)` | `{ anyOf: [...] }` |
| `intersect(...)` | `{ allOf: [...] }` |
| `literal(v)` | `{ const: v }` |
| `optional(T)` | T (not in required) |
| `nullable(T)` | `{ type: [T, "null"] }` |

## Supported Constraints

| Vex Constraint | JSON Schema |
|----------------|-------------|
| `min(n)` / `minLength` | `minLength` |
| `max(n)` / `maxLength` | `maxLength` |
| `len(n)` | `minLength` + `maxLength` |
| `pattern(regex)` | `pattern` |
| `email` | `format: "email"` |
| `url` | `format: "uri"` |
| `uuid` | `format: "uuid"` |
| `datetime` | `format: "date-time"` |
| `dateOnly` | `format: "date"` |
| `time` | `format: "time"` |
| `ipv4` | `format: "ipv4"` |
| `ipv6` | `format: "ipv6"` |
| `int` | `type: "integer"` |
| `gte(n)` | `minimum` |
| `gt(n)` | `exclusiveMinimum` |
| `lte(n)` | `maximum` |
| `lt(n)` | `exclusiveMaximum` |
| `positive` | `exclusiveMinimum: 0` |
| `negative` | `exclusiveMaximum: 0` |
| `multipleOf(n)` | `multipleOf` |

## License

MIT
