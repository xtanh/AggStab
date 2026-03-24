# @sylphx/vex

> ⚡ Ultra-fast schema validation - 12x faster than Zod, 6x faster than Valibot

## Installation

```bash
npm install @sylphx/vex
# or
bun add @sylphx/vex
# or
pnpm add @sylphx/vex
```

## Basic Usage

```typescript
import { str, num, object, array, email, int, positive, min, max, safeParse } from '@sylphx/vex'

// Create a schema
const userSchema = object({
  name: str(min(1), max(100)),
  email: str(email),
  age: num(int, positive),
})

// Validate data (throws on error)
const user = userSchema({ name: 'Alice', email: 'alice@example.com', age: 30 })

// Or use safeParse (returns result object)
const result = safeParse(userSchema)({ name: 'Bob', email: 'invalid', age: -1 })
if (result.success) {
  console.log(result.data)
} else {
  console.log(result.error) // "email: Invalid email"
}
```

## Core Concepts

### 1. Type Validators

Vex provides composable type validators:

```typescript
import { str, num, bool, arr, obj } from '@sylphx/vex'

// Base validators (no constraints)
str()('hello')      // ✅ 'hello'
str()(123)          // ❌ throws "Expected string"

num()(42)           // ✅ 42
num()('42')         // ❌ throws "Expected number"

bool()(true)        // ✅ true
```

### 2. Adding Constraints

Pass constraints directly to type validators:

```typescript
import { str, num, email, url, int, positive, min, max, gte, lte } from '@sylphx/vex'

// String with constraints
str(email)                    // string + email format
str(url)                      // string + URL format
str(min(3), max(20))          // string + length between 3-20

// Number with constraints
num(int)                      // integer
num(positive)                 // positive number
num(int, positive)            // positive integer
num(gte(0), lte(100))         // number between 0-100

// Usage
str(email)('test@example.com')    // ✅ 'test@example.com'
str(email)('invalid')             // ❌ throws "Invalid email"

num(int, positive)(25)            // ✅ 25
num(int, positive)(-5)            // ❌ throws "Must be positive"
```

### 3. Object Schemas

```typescript
import { object, str, num, email, int, positive, optional, min } from '@sylphx/vex'

const userSchema = object({
  id: str(),
  name: str(min(1)),
  email: str(email),
  age: num(int, positive),
  bio: optional(str()),  // optional field
})

// TypeScript infers the type automatically
type User = ReturnType<typeof userSchema>
// { id: string; name: string; email: string; age: number; bio?: string }

const user = userSchema({
  id: '123',
  name: 'Alice',
  email: 'alice@example.com',
  age: 30,
})
```

### 4. Array Schemas

```typescript
import { array, str, num, email, int, nonempty } from '@sylphx/vex'

const emailsSchema = array(str(email))
const scoresSchema = array(num(int))
const tagsSchema = array(str(nonempty))

emailsSchema(['a@b.com', 'c@d.com'])  // ✅
scoresSchema([1, 2, 3])               // ✅
tagsSchema(['a', '', 'c'])            // ❌ throws "[1]: Required"
```

### 5. Error Handling

Three ways to handle validation errors:

```typescript
import { str, email, safeParse, tryParse, ValidationError } from '@sylphx/vex'

const schema = str(email)

// Method 1: Direct call (throws on error)
try {
  schema('invalid')
} catch (e) {
  if (e instanceof ValidationError) {
    console.log(e.message) // "Invalid email"
  }
}

// Method 2: safeParse (returns result object)
const result = safeParse(schema)('invalid')
if (!result.success) {
  console.log(result.error) // "Invalid email"
}

// Method 3: tryParse (returns null on error)
const value = tryParse(schema)('invalid') // null
```

## Common Patterns

### Form Validation

```typescript
import { object, str, num, email, min, max, int, positive, safeParse } from '@sylphx/vex'

const signupSchema = object({
  username: str(min(3), max(20)),
  email: str(email),
  password: str(min(8)),
  age: num(int, positive),
})

function handleSubmit(formData: unknown) {
  const result = safeParse(signupSchema)(formData)

  if (!result.success) {
    return { error: result.error }
  }

  // result.data is fully typed
  return { user: result.data }
}
```

### API Response Validation

```typescript
import { object, array, str, num, bool, int, positive } from '@sylphx/vex'

const apiResponseSchema = object({
  status: str(),
  data: array(object({
    id: num(int, positive),
    title: str(),
    completed: bool(),
  })),
})

async function fetchTodos() {
  const response = await fetch('/api/todos')
  const json = await response.json()

  // Throws if response doesn't match schema
  return apiResponseSchema(json)
}
```

### Optional & Nullable

```typescript
import { str, email, optional, nullable, nullish, withDefault } from '@sylphx/vex'

// optional - allows undefined
const optionalEmail = optional(str(email))
optionalEmail(undefined)           // ✅ undefined
optionalEmail('test@example.com')  // ✅ 'test@example.com'

// nullable - allows null
const nullableEmail = nullable(str(email))
nullableEmail(null)                // ✅ null

// nullish - allows null or undefined
const nullishEmail = nullish(str(email))

// withDefault - provides default value
const emailWithDefault = withDefault(str(email), 'default@example.com')
emailWithDefault(undefined)        // ✅ 'default@example.com'
```

### Union Types

```typescript
import { union, str, num, literal } from '@sylphx/vex'

// Union of types (variadic syntax)
const stringOrNumber = union(str(), num())
stringOrNumber('hello')  // ✅ 'hello'
stringOrNumber(42)       // ✅ 42
stringOrNumber(true)     // ❌ throws

// Literal union (discriminated)
const status = union(literal('pending'), literal('active'), literal('done'))
status('active')         // ✅ 'active'
status('invalid')        // ❌ throws
```

### Transforms

```typescript
import { str, num, trim, lower, upper, toInt, toFloat, toDate, pipe } from '@sylphx/vex'

// String transforms (use pipe for transforms)
const normalizedEmail = pipe(str(), trim, lower)
normalizedEmail('  ALICE@EXAMPLE.COM  ')  // 'alice@example.com'

// Parse string to number
const parseAge = pipe(str(), toInt)
parseAge('42')  // 42

// Parse string to Date
const parseDate = pipe(str(), toDate)
parseDate('2024-01-15')  // Date object
```

## API Reference

### Type Validators

| Validator | Description |
|-----------|-------------|
| `str(...constraints)` | String validator |
| `num(...constraints)` | Number validator (excludes NaN) |
| `bool()` | Boolean validator |
| `arr()` | Array validator |
| `obj()` | Object validator |
| `bigInt()` | BigInt validator |
| `date()` | Date validator |

### String Constraints

| Constraint | Description |
|------------|-------------|
| `email` | Email format |
| `url` | URL format |
| `uuid` | UUID format |
| `min(n)` | Minimum length |
| `max(n)` | Maximum length |
| `len(n)` | Exact length |
| `nonempty` | Non-empty string |
| `pattern(regex)` | Regex pattern |
| `startsWith(s)` | Starts with string |
| `endsWith(s)` | Ends with string |
| `includes(s)` | Contains string |

### Number Constraints

| Constraint | Description |
|------------|-------------|
| `int` | Integer |
| `positive` | > 0 |
| `negative` | < 0 |
| `finite` | Finite number |
| `gte(n)` | >= n |
| `lte(n)` | <= n |
| `gt(n)` | > n |
| `lt(n)` | < n |
| `multipleOf(n)` | Divisible by n |

### Transforms

| Transform | Description |
|-----------|-------------|
| `trim` | Trim whitespace |
| `lower` | Lowercase |
| `upper` | Uppercase |
| `toInt` | Parse to integer |
| `toFloat` | Parse to float |
| `toDate` | Parse to Date |

### Composition

| Function | Description |
|----------|-------------|
| `object(shape, ...meta)` | Object schema |
| `array(schema, ...meta)` | Array schema |
| `tuple(...schemas, ...meta)` | Tuple schema |
| `union(...schemas, ...meta)` | Union type |
| `intersect(...schemas, ...meta)` | Intersection type |
| `optional(schema)` | Allow undefined |
| `nullable(schema)` | Allow null |
| `nullish(schema)` | Allow null or undefined |
| `withDefault(schema, value)` | Default value |
| `pipe(...validators)` | Chain validators (for transforms) |

### Metadata

Metadata modifiers return `MetaAction` and are passed directly to schema functions:

```typescript
import { str, num, object, union, description, title, examples } from '@sylphx/vex'

// Add metadata to primitives
str(description('User name'))
str(min(1), description('Required name'), title('Name'))
num(int, positive, examples([1, 2, 3]))

// Add metadata to composites
union(str(), num(), description('String or number'))
object({ name: str() }, description('User object'))
array(str(), description('List of names'))
tuple(str(), num(), description('Name and age pair'))
```

| Function | Description |
|----------|-------------|
| `description(text)` | Add description (returns MetaAction) |
| `title(text)` | Add title (returns MetaAction) |
| `examples(values)` | Add examples (returns MetaAction) |
| `deprecated()` | Mark as deprecated (returns MetaAction) |
| `metadata(obj)` | Add multiple metadata fields (returns MetaAction) |
| `brand(schema, name)` | Nominal typing (strict) - wraps schema |
| `flavor(schema, name)` | Nominal typing (weak) - wraps schema |
| `readonly(schema)` | Mark as readonly - wraps schema |
| `getDescription(schema)` | Get description |
| `getTitle(schema)` | Get title |
| `getExamples(schema)` | Get examples |
| `getMeta(schema)` | Get all metadata |

### Utilities

| Function | Description |
|----------|-------------|
| `safeParse(schema)(data)` | Returns `{ success, data/error }` |
| `tryParse(schema)(data)` | Returns data or null |
| `toJsonSchema(schema)` | Convert to JSON Schema |

## Standard Schema

Vex implements [Standard Schema](https://standardschema.dev/) v1, compatible with:
- tRPC
- TanStack Form/Router
- Hono
- Remix
- And more...

```typescript
// All validators expose ~standard property
const schema = object({ email: str(email) })
schema['~standard'].validate(data)
```

## JSON Schema

Convert Vex schemas to JSON Schema (draft-07, draft-2019-09, draft-2020-12):

```typescript
import { object, str, num, email, int, positive, optional, toJsonSchema } from '@sylphx/vex'

const userSchema = object({
  name: str(),
  email: str(email),
  age: num(int, positive),
  bio: optional(str()),
})

// Convert to JSON Schema
const jsonSchema = toJsonSchema(userSchema)
// {
//   "$schema": "http://json-schema.org/draft-07/schema#",
//   "type": "object",
//   "properties": {
//     "name": { "type": "string" },
//     "email": { "type": "string", "format": "email" },
//     "age": { "type": "integer", "exclusiveMinimum": 0 },
//     "bio": { "type": "string" }
//   },
//   "required": ["name", "email", "age"]
// }
```

### Options

```typescript
// Choose JSON Schema draft version
toJsonSchema(schema, { draft: 'draft-2020-12' })

// Without $schema property
toJsonSchema(schema, { $schema: false })

// With named definitions
toJsonSchema(schema, {
  definitions: {
    User: userSchema,
    Post: postSchema,
  }
})
```

### Use Cases

- **OpenAPI/Swagger**: Generate API documentation
- **Form builders**: Auto-generate forms from schemas
- **Code generation**: Generate types for other languages
- **Validation interop**: Share schemas with non-JS systems

## Schema Metadata

Add documentation metadata to your schemas using MetaAction modifiers:

```typescript
import { str, num, object, union, description, title, examples, toJsonSchema, getMeta } from '@sylphx/vex'

// Add metadata directly in schema definitions
const emailSchema = str(
  email,
  title('Email'),
  description('User email address'),
  examples(['user@example.com', 'admin@test.com'])
)

// Works with all schema types
const userSchema = object(
  {
    name: str(min(1), description('User full name')),
    email: emailSchema,
    age: num(int, positive, description('User age in years')),
  },
  description('User account object'),
  title('User')
)

// Works with unions too
const idSchema = union(
  str(),
  num(),
  description('Identifier can be string or number')
)

// Metadata flows to JSON Schema
toJsonSchema(emailSchema)
// {
//   "type": "string",
//   "format": "email",
//   "title": "Email",
//   "description": "User email address",
//   "examples": ["user@example.com", "admin@test.com"]
// }
```

### Nominal Typing

Use `brand` for strict nominal types or `flavor` for weak nominal types:

```typescript
import { str, uuid, brand, flavor, pipe } from '@sylphx/vex'

// Strict brand - types are incompatible
type UserId = string & { __brand: 'UserId' }
const userId = brand(pipe(str(), uuid), 'UserId')

// Weak flavor - types are compatible but distinguishable
type Email = string & { __flavor?: 'Email' }
const emailFlavor = flavor(str(email), 'Email')
```

## Performance

| Benchmark | Vex | Zod | Valibot | vs Zod | vs Valibot |
|-----------|-----|-----|---------|--------|------------|
| string | 367M | 50M | 50M | 7.4x | 7.3x |
| email | 128M | 12M | 17M | 11x | 7.5x |
| url | 110M | 4M | 5M | 28x | 23x |
| uuid | 117M | 16M | 6M | 7.3x | 19x |
| object (3 fields) | 18M | 5M | 7M | 3.9x | 2.7x |
| object (nested) | 11M | 5M | 5M | 2x | 2.1x |
| array[50] | 8.5M | 780K | 1.4M | 11x | 6x |
| safeParse (valid) | 15M | 7M | 6M | 2.1x | 2.5x |
| safeParse (invalid) | 44M | 376K | 2.6M | 118x | 17x |

**Average: 12x faster than Zod, 6x faster than Valibot**

## License

MIT
