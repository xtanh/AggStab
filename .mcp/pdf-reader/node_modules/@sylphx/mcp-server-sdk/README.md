# @sylphx/mcp-server-sdk

Pure functional MCP (Model Context Protocol) server SDK for Bun.

[![npm](https://img.shields.io/npm/v/@sylphx/mcp-server-sdk)](https://www.npmjs.com/package/@sylphx/mcp-server-sdk)
[![MCP Conformance](https://img.shields.io/badge/MCP%20Conformance-88%25-brightgreen)](https://github.com/modelcontextprotocol/conformance)

## Features

- **Pure Functional**: Immutable data, composable handlers
- **Type-Safe**: First-class TypeScript with Vex schema integration
- **Builder Pattern**: Fluent API for defining tools, resources, and prompts
- **Fast**: Built for Bun with minimal dependencies
- **Streamable HTTP**: MCP 2025-03-26 spec with SSE notifications
- **Complete**: Tools, resources, prompts, notifications, sampling, elicitation

## Installation

```bash
bun add @sylphx/mcp-server-sdk
```

## Quick Start

```typescript
import { createServer, tool, text, stdio } from "@sylphx/mcp-server-sdk"
import { object, str } from "@sylphx/vex"

// Define tools using builder pattern
const greet = tool()
  .description("Greet someone")
  .input(object({ name: str() }))
  .handler(({ input }) => text(`Hello, ${input.name}!`))

const ping = tool()
  .handler(() => text("pong"))

// Create and start server
const server = createServer({
  name: "my-server",
  version: "1.0.0",
  tools: { greet, ping },
  transport: stdio()
})

await server.start()
```

## Tools

Tools are callable functions exposed to the AI.

```typescript
import { tool, text, image, audio, json, toolError } from "@sylphx/mcp-server-sdk"
import { description, enum_, num, object, str } from "@sylphx/vex"

// Simple tool (no input)
const ping = tool()
  .description("Health check")
  .handler(() => text("pong"))

// Tool with typed input
const calculator = tool()
  .description("Perform arithmetic")
  .input(object({
    a: num(description("First number")),
    b: num(description("Second number")),
    op: enum_(["+", "-", "*", "/"] as const),
  }))
  .handler(({ input }) => {
    const { a, b, op } = input
    const result = op === "+" ? a + b
      : op === "-" ? a - b
      : op === "*" ? a * b
      : a / b
    return text(`${a} ${op} ${b} = ${result}`)
  })

// Multiple content items
const systemInfo = tool()
  .description("Get system information")
  .handler(() => [
    text("CPU: 8 cores"),
    text("Memory: 16GB")
  ])

// Mixed content types
const screenshot = tool()
  .description("Take screenshot with description")
  .handler(() => [
    text("Here's the screenshot:"),
    image(base64Data, "image/png")
  ])

// Return JSON data
const getUser = tool()
  .description("Get user data")
  .input(object({ id: str() }))
  .handler(({ input }) => json({ id: input.id, name: "Alice" }))

// Return error
const riskyOperation = tool()
  .description("May fail")
  .handler(() => toolError("Something went wrong"))
```

## Resources

Resources provide data to the AI.

```typescript
import { resource, resourceTemplate, resourceText, resourceBlob } from "@sylphx/mcp-server-sdk"

// Static resource with fixed URI
const readme = resource()
  .uri("file:///readme.md")
  .description("Project readme")
  .mimeType("text/markdown")
  .handler(({ uri }) => resourceText(uri, "# My Project\n\nWelcome!"))

// Resource template for dynamic URIs
const fileReader = resourceTemplate()
  .uriTemplate("file:///{path}")
  .description("Read any file")
  .handler(async ({ uri, params }) => {
    const content = await Bun.file(`/${params.path}`).text()
    return resourceText(uri, content)
  })

// Binary resource
const logo = resource()
  .uri("image:///logo.png")
  .mimeType("image/png")
  .handler(async ({ uri }) => {
    const data = await Bun.file("./logo.png").bytes()
    const base64 = Buffer.from(data).toString("base64")
    return resourceBlob(uri, base64, "image/png")
  })
```

## Prompts

Prompts are reusable conversation templates.

```typescript
import { prompt, user, assistant, messages, promptResult } from "@sylphx/mcp-server-sdk"
import { description, object, optional, str, withDefault } from "@sylphx/vex"

// Simple prompt (no arguments)
const greeting = prompt()
  .description("A friendly greeting")
  .handler(() => messages(
    user("Hello!"),
    assistant("Hi there! How can I help you today?")
  ))

// Prompt with typed arguments
const codeReview = prompt()
  .description("Review code for issues")
  .args(object({
    code: str(description("Code to review")),
    language: optional(str(description("Programming language"))),
  }))
  .handler(({ args }) => messages(
    user(`Please review this ${args.language ?? "code"}:\n\`\`\`\n${args.code}\n\`\`\``),
    assistant("I'll analyze this code for potential issues, best practices, and improvements.")
  ))

// Prompt with description in result
const translate = prompt()
  .description("Translate text between languages")
  .args(object({
    text: str(),
    from: withDefault(str(), "auto"),
    to: str(),
  }))
  .handler(({ args }) => promptResult(
    `Translation from ${args.from} to ${args.to}`,
    messages(user(`Translate "${args.text}" from ${args.from} to ${args.to}`))
  ))
```

## Server Configuration

```typescript
import { createServer, stdio, http } from "@sylphx/mcp-server-sdk"

const server = createServer({
  // Server identity
  name: "my-server",
  version: "1.0.0",
  instructions: "This server provides...",

  // Handlers (names from object keys)
  tools: { greet, ping, calculator },
  resources: { readme, config },
  resourceTemplates: { file: fileReader },
  prompts: { codeReview, translate },

  // Transport
  transport: stdio()  // or http({ port: 3000 })
})

await server.start()
```

## Transports

### Stdio Transport

For CLI tools and subprocess communication.

```typescript
import { stdio } from "@sylphx/mcp-server-sdk"

const server = createServer({
  tools: { ping },
  transport: stdio()
})

await server.start()
```

### HTTP Transport

Implements MCP Streamable HTTP (2025-03-26 spec) with SSE support for real-time notifications.

```typescript
import { http } from "@sylphx/mcp-server-sdk"

const server = createServer({
  tools: { ping },
  transport: http({
    port: 3000,
    cors: "*"  // Enable CORS for web clients
  })
})

await server.start()
// Server running at http://localhost:3000/mcp
```

**Endpoints:**
- `POST /mcp` - JSON-RPC messages (returns JSON or SSE based on Accept header)
- `GET /mcp/health` - Health check

**SSE Streaming:**

When client sends `Accept: text/event-stream`, the server responds with SSE format, enabling real-time notifications during request processing:

```
event: message
data: {"jsonrpc":"2.0","method":"notifications/progress","params":{...}}

event: message
data: {"jsonrpc":"2.0","id":1,"result":{...}}
```

## Notifications

Send server-to-client notifications for progress and logging using the simplified context API.

```typescript
import { array, object, str } from "@sylphx/vex"

const processFiles = tool()
  .description("Process multiple files")
  .input(object({ files: array(str()) }))
  .handler(async ({ input, ctx }) => {
    const total = input.files.length

    for (let i = 0; i < total; i++) {
      // Report progress (automatically uses progressToken from request)
      ctx.progress(i + 1, { total, message: `Processing ${input.files[i]}` })
      await processFile(input.files[i])
    }

    // Log completion
    ctx.log("info", { message: "Processing complete" })

    return text(`Processed ${total} files`)
  })
```

### Context Methods

```typescript
// Progress notification (uses progressToken from request automatically)
ctx.progress(current: number, options?: { total?: number; message?: string })

// Log notification
ctx.log(level: "debug" | "info" | "notice" | "warning" | "error" | "critical" | "alert" | "emergency", data: unknown, logger?: string)
```

### Low-level Notification Factories

For advanced use cases, you can use the raw notification factories:

```typescript
import { progress, log, resourcesListChanged, toolsListChanged, promptsListChanged, resourceUpdated, cancelled } from "@sylphx/mcp-server-sdk"

// Progress notification (requires manual token)
progress(token: string | number, current: number, options?: { total?: number; message?: string })

// Log notification
log(level: LogLevel, data: unknown, logger?: string)

// List change notifications (for dynamic capability updates)
resourcesListChanged()
toolsListChanged()
promptsListChanged()

// Resource updated notification
resourceUpdated(uri: string)

// Cancellation notification
cancelled(requestId: string | number, reason?: string)
```

## Sampling

Request LLM completions from the client.

```typescript
import { createSamplingClient } from "@sylphx/mcp-server-sdk"
import { object, str } from "@sylphx/vex"

const summarize = tool()
  .description("Summarize text using AI")
  .input(object({ text: str() }))
  .handler(async ({ input, ctx }) => {
    const sampling = createSamplingClient(ctx.requestSampling)

    const result = await sampling.createMessage({
      messages: [
        { role: "user", content: { type: "text", text: `Summarize: ${input.text}` } }
      ],
      maxTokens: 500,
      // Optional parameters
      systemPrompt: "You are a helpful summarizer",
      temperature: 0.7,
      stopSequences: ["END"],
      modelPreferences: {
        hints: [{ name: "claude-3" }],
        costPriority: 0.5,
        speedPriority: 0.5,
        intelligencePriority: 0.8,
      },
    })

    // result.content is the response content
    // result.model is the model used
    // result.stopReason is why generation stopped
    return text(result.content.text)
  })
```

## Elicitation

Request user input from the client.

```typescript
import { createElicitationClient } from "@sylphx/mcp-server-sdk"
import { object, str } from "@sylphx/vex"

const confirmAction = tool()
  .description("Confirm before proceeding")
  .input(object({ action: str() }))
  .handler(async ({ input, ctx }) => {
    const elicit = createElicitationClient(ctx.requestElicitation)

    const result = await elicit.elicit(
      `Are you sure you want to ${input.action}?`,
      {
        type: "object",
        properties: {
          confirm: {
            type: "boolean",
            description: "Confirm action",
          },
          reason: {
            type: "string",
            description: "Optional reason",
          },
        },
        required: ["confirm"],
      }
    )

    // result.action: "accept" | "decline" | "cancel"
    // result.content: { confirm: boolean, reason?: string } (when action is "accept")

    if (result.action === "accept" && result.content?.confirm) {
      return text(`Proceeding with ${input.action}`)
    }

    return text("Action cancelled")
  })
```

### Elicitation Schema Properties

```typescript
interface ElicitationProperty {
  type: "string" | "number" | "integer" | "boolean"
  description?: string
  default?: string | number | boolean
  enum?: (string | number)[]        // Constrain to specific values
  enumNames?: string[]              // Display names for enum values
  // String-specific
  format?: "email" | "uri" | "date" | "date-time"
  minLength?: number
  maxLength?: number
  // Number-specific
  minimum?: number
  maximum?: number
}
```

## Pagination

Paginate large result sets.

```typescript
import { paginate } from "@sylphx/mcp-server-sdk"
import { object, optional, str } from "@sylphx/vex"

const listItems = tool()
  .description("List items with pagination")
  .input(object({ cursor: optional(str()) }))
  .handler(async ({ input }) => {
    const allItems = await fetchAllItems()

    const result = paginate(allItems, input.cursor, {
      defaultPageSize: 10,
      maxPageSize: 100,
    })

    // result.items: current page items
    // result.nextCursor: cursor for next page (undefined if last page)
    return json({
      items: result.items,
      nextCursor: result.nextCursor,
    })
  })
```

## API Reference

### Server

```typescript
createServer(config: ServerConfig): Server

interface ServerConfig {
  name?: string                    // Default: "mcp-server"
  version?: string                 // Default: "1.0.0"
  instructions?: string            // Instructions for the LLM
  tools?: Record<string, ToolDefinition>
  resources?: Record<string, ResourceDefinition>
  resourceTemplates?: Record<string, ResourceTemplateDefinition>
  prompts?: Record<string, PromptDefinition>
  transport: TransportFactory
}
```

### Tool Builder

```typescript
tool()
  .description(string)                    // Optional description
  .input(VexSchema)                       // Optional input schema
  .handler(fn: HandlerFn) -> ToolDefinition

// Handler signature
({ input, ctx }) => ToolResult | Promise<ToolResult>

// Handler can return:
// - Single content:  text("hello")
// - Array:           [text("hi"), image(data, "image/png")]
// - Full result:     { content: [...], isError: true }
```

### Resource Builder

```typescript
resource()
  .uri(string)                            // Required URI
  .description(string)                    // Optional description
  .mimeType(string)                       // Optional MIME type
  .handler(fn) -> ResourceDefinition

resourceTemplate()
  .uriTemplate(string)                    // Required URI template (RFC 6570)
  .description(string)                    // Optional description
  .mimeType(string)                       // Optional MIME type
  .handler(fn) -> ResourceTemplateDefinition

// Handler receives { uri, ctx } or { uri, params, ctx }
```

### Prompt Builder

```typescript
prompt()
  .description(string)                    // Optional description
  .args(VexSchema)                        // Optional arguments schema
  .handler(fn) -> PromptDefinition

// Handler receives { args, ctx } or { ctx }
```

### Content Helpers

```typescript
// Tool content
text(content: string, annotations?): TextContent
image(data: string, mimeType: string, annotations?): ImageContent
audio(data: string, mimeType: string, annotations?): AudioContent
embedded(resource: EmbeddedResource, annotations?): ResourceContent
json(data: unknown): TextContent
toolError(message: string): ToolsCallResult

// Resources
resourceText(uri: string, text: string, mimeType?: string): ResourcesReadResult
resourceBlob(uri: string, blob: string, mimeType: string): ResourcesReadResult
resourceContents(...items: EmbeddedResource[]): ResourcesReadResult

// Prompts
user(content: string): PromptMessage
assistant(content: string): PromptMessage
messages(...msgs: PromptMessage[]): PromptsGetResult
promptResult(description: string, result: PromptsGetResult): PromptsGetResult
```

### Transports

```typescript
stdio(options?: StdioOptions): TransportFactory
http(options?: HttpOptions): TransportFactory

interface HttpOptions {
  port?: number        // Default: 3000
  hostname?: string    // Default: "localhost"
}
```

### Notifications

```typescript
progress(token, current, options?): ProgressNotification
log(level, data, logger?): LogNotification
resourcesListChanged(): Notification
toolsListChanged(): Notification
promptsListChanged(): Notification
resourceUpdated(uri): Notification
cancelled(requestId, reason?): Notification
```

### Sampling

```typescript
createSamplingClient(sender: SamplingRequestSender): SamplingClient

interface SamplingClient {
  createMessage(params: SamplingCreateParams): Promise<SamplingCreateResult>
}
```

### Elicitation

```typescript
createElicitationClient(sender: ElicitationRequestSender): ElicitationClient

interface ElicitationClient {
  elicit(message: string, schema: ElicitationSchema): Promise<ElicitationCreateResult>
}
```

### Pagination

```typescript
paginate<T>(items: T[], cursor?: string, options?: PaginationOptions): PageResult<T>

interface PaginationOptions {
  defaultPageSize?: number   // Default: 50
  maxPageSize?: number       // Default: 100
}

interface PageResult<T> {
  items: T[]
  nextCursor?: string
}
```

## Powered by Sylphx

- [@sylphx/vex](https://github.com/SylphxAI/vex) - Ultra-fast schema validation
- [@sylphx/gust](https://github.com/SylphxAI/gust) - Lightweight HTTP framework for Bun
- [@sylphx/biome-config](https://github.com/SylphxAI/biome-config) - Shared Biome configuration
- [@sylphx/tsconfig](https://github.com/SylphxAI/tsconfig) - Shared TypeScript configuration
- [@sylphx/doctor](https://github.com/SylphxAI/doctor) - Project health checker
- [@sylphx/bump](https://github.com/SylphxAI/bump) - Version management

## License

MIT
