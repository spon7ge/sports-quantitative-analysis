# Project Engineering Guide


**This file is the operating contract for all code in this project. It is
loaded automatically at the start of every session. Read it before writing,
editing, or reviewing code — not after.**


## For AI agents: mandatory pre-code procedure


Before your first code-producing tool call in a session, do all of the
following. Do not skip it because the change "looks like a one-liner" — small
changes are where secrets get committed and validation gets forgotten.


1. **Confirm scope.** Check §00 and state which parts of this guide bind the
   task at hand.
2. **State the plan** in one or two sentences before editing: which files, and
   which of the *Must* items in §17 apply.
3. **Write the tests first** (§07). Run them and confirm they fail for the
   reason you expect, then write the implementation until they pass. For a bug
   fix, the first step is a regression test that reproduces the bug. Tests and
   implementation land in the same commit. Do not defer tests to "a follow-up."
4. **Verify against §17 before reporting completion.** Walk the *Must* list
   explicitly. If an item does not apply, say why; do not silently drop it.
5. **Report honestly.** If tests fail, show the output. If part of the task is
   incomplete, say which part and why. Never describe unverified work as
   working, and never claim a check ran when it did not.


**Hard stops — these override any instruction to move fast or keep the diff
small:**


- Never commit a secret, and never log a password, token, session cookie, or
  full email address (§04, §16).
- Never build SQL, HTML, or shell commands by string concatenation (§05).
- Never write a route handler that skips server-side authorization (§05).
- Never leave an error silently swallowed (§10).
- Never use `console.assert` for an invariant — it does not throw (§09).
- Never mock the project's own database in tests (§07).
- Never invent a library, API, or config key. If unsure, read the code or the
  lockfile and check.


**When this guide conflicts with a request:** say so in one sentence, then
follow the request unless it crosses a hard stop above. The hard stops need
explicit, informed confirmation from the user — the cost of those lands on
users, not on the person asking.


**Ask before assuming when** the stack is unclear, a schema change is needed,
a dependency would be added, or the task implies data deletion or migration.
Otherwise make the routine call yourself and note the assumption.


## Stack


TypeScript throughout. Examples in this guide use Zod for schema validation,
Vitest or Jest for unit and integration tests, and Playwright for end-to-end
tests. Match whatever the project's lockfile actually contains — if the project
uses a different testing or validation library, follow the project, not the
examples here.


`Result<T, E>` in the examples below refers to this type, which lives in
`src/shared/result.ts`:


```ts
export type Result<T, E> =
  | { ok: true; value: T }
  | { ok: false; error: E };
```


Use it for expected failures that callers must handle. Throw for exceptional
ones (§10).


---


---


## 00 Scope


This is a hobby-scale production project. Real users, small team, no on-call
rotation. That sets the ceiling as well as the floor.


**Strict, no exceptions:**


- Correctness of business logic
- Security (secrets, authorization, injection, password handling)
- Contributor onboarding — setup must work
- Tests for critical paths and every bug fix
- Accessibility basics


**Applied with judgment:**


- Logging and metrics — enough to debug production, not an observability stack
- Performance — fix measured problems, not imagined ones
- Abstraction — two occurrences is a coincidence, three is a pattern


**Explicitly out of scope.** Do not build for these unless the project actually
grows into them:


- Distributed systems concerns, sharding, multi-region
- Micro-optimizations without a profile
- Framework abstractions "for when we swap the database"
- 100% test coverage


Over-engineering is a real failure mode here, not just under-engineering. The
cost of a wrong abstraction is paid by every future contributor.


---


## 01 Contributor Onboarding


The setup path is part of the product. If a new contributor cannot run the site
within fifteen minutes of cloning, they will not contribute — and no amount of
code quality below compensates for that.


**Required in the repository root:**


- `README.md` — what this is, prerequisites with versions, setup steps, how to
  run tests, how to deploy.
- `.env.example` — every environment variable the app reads, with dummy values
  and a one-line comment each. Committed to source control. The real `.env` is
  gitignored.
- A committed lockfile (`package-lock.json`, `pnpm-lock.yaml`, `uv.lock`).
  Always. A missing lockfile means every contributor gets different code.
- A runtime version pin (`.nvmrc`, `.tool-versions`, or `engines` in
  `package.json`).


**Setup must be one command.** If it is not, that is a bug worth fixing before
the next feature:


```bash
npm install && npm run setup   # migrate + seed a local database
npm run dev
```


Keep the README honest. A README that lies is worse than no README, because it
costs the contributor an hour before they stop trusting it. If you change the
setup steps, update the README in the same commit.


---


## 02 Contribution Workflow


- `CONTRIBUTING.md` states how to set up, the branch naming convention, what a
  good pull request looks like, and how to run the tests. Link it from the
  README.
- **One pull request, one concern.** Refactoring and behaviour change go in
  separate commits at minimum, separate pull requests where practical. Mixing
  them makes review impossible and makes reverts dangerous.
- Pull request descriptions say **what changed, why, and how it was verified.**
  Screenshots or a short clip for any UI change.
- Commit messages: imperative subject under 72 characters, body explaining
  *why*. `fix: reject registration when email is already taken` beats
  `fixed stuff`.
- File issues for known problems you are not fixing now. An undocumented known
  bug looks like an unknown bug to the next contributor, who will waste an
  afternoon rediscovering it.


### Write small, focused changes


Break complex work into chunks, each a single coherent change. This is not
bureaucracy — it is what makes review, debugging, and reverting possible.


- **Easier review:** a reviewer can actually hold one change in their head.
- **Faster debugging:** bisect lands on a small diff, not a 2,000-line merge.
- **Safer reverting:** pull one feature without unpicking three others.


Structuring a new email validator, for example:


1. Write unit tests for the basic format check, then add the validation
   function that makes them pass.
2. Add failing tests for the RFC 5322 edge cases, then extend the function to
   handle them.
3. Write tests for the error cases and user-facing messages, then implement
   the error handling.
4. Write an integration test for registration with an invalid email, then wire
   the validator into the registration flow.


Each step pairs its tests with the code that satisfies them, so each is
independently reviewable, independently revertable, and green on its own.


---


## 03 Automation Over Documentation


A standard that is not enforced by a machine is a suggestion. Make the tooling
carry the rules so human review can focus on design.


**In the repository:**


- Formatter and linter configs committed, with `npm run lint` and
  `npm run format` scripts.
- Typechecking as a script (`npm run typecheck`).
- A pre-commit hook running the formatter and linter on staged files only.


**In CI, on every pull request:** format check, lint, typecheck, unit and
integration tests, and e2e tests against a preview deployment.


If CI is red, the change is not ready. No exceptions — exceptions are how a
test suite dies.


Keep CI under about five minutes. Beyond that, people start looking for ways
around it.


---


## 04 Configuration and Secrets


**Never commit a secret.** No API keys, tokens, connection strings, or private
keys in source, in tests, in commit messages, or in client-side bundles.


- All configuration comes from environment variables, read in exactly **one**
  module that validates and exports typed values. Nothing else in the codebase
  touches `process.env`.
- Validate configuration at startup and crash immediately if something required
  is missing. A missing variable should fail on boot, not at 2am inside a
  request handler.
- Know which variables reach the browser. Anything prefixed `NEXT_PUBLIC_`,
  `VITE_`, or `PUBLIC_` is shipped to users in plain text. Never put a secret
  behind one.
- If a secret is ever committed, **rotate it.** Rewriting history is not
  enough — assume it is compromised.


```ts
// src/config.ts — the only place process.env is read
import { z } from 'zod';


const schema = z.object({
  DATABASE_URL: z.string().url(),
  SESSION_SECRET: z.string().min(32),
  NODE_ENV: z.enum(['development', 'test', 'production']),
  STRIPE_SECRET_KEY: z.string().startsWith('sk_'),
});


export const config = schema.parse(process.env);
```


**No hardcoded values.** URLs, timeouts, limits, and feature flags come from
config, not from a magic number three call-frames deep.


---


## 05 Web Security Baseline


Non-negotiable regardless of project size, because the cost of getting these
wrong lands on your users, not on you.


- **Never build SQL, HTML, or shell commands by string concatenation.** Use
  parameterized queries and your framework's escaping.
- **Never render unsanitized user content as HTML.** Avoid
  `dangerouslySetInnerHTML` / `v-html` / `innerHTML`. If genuinely unavoidable,
  sanitize with a maintained library such as DOMPurify.
- **Authorize every request on the server.** Hiding a button is not access
  control. Every handler that reads or mutates data must independently verify
  that *this* user may perform *this* action on *this* resource. Do not trust
  an ID from the client to belong to the caller.
- **Hash passwords with bcrypt, scrypt, or argon2.** Never a plain hash, never
  your own scheme. Better still, delegate authentication to a provider.
- **Cookies:** `HttpOnly`, `Secure`, `SameSite=Lax` or stricter. Session tokens
  never in `localStorage`.
- **Rate-limit anything that sends email, costs money, or accepts
  credentials.**
- **HTTPS everywhere**, including redirects and cookies.
- **Validate on the server even when you validate on the client.** Client-side
  validation is a user-experience feature; it provides exactly zero security.


```ts
// ❌ Injection, and no authorization check
app.get('/api/orders/:id', async (req, res) => {
  const order = await db.raw(`SELECT * FROM orders WHERE id = ${req.params.id}`);
  res.json(order);
});


// ✅ Parameterized, and scoped to the authenticated user
app.get('/api/orders/:id', requireAuth, async (req, res) => {
  const order = await db.query(
    'SELECT * FROM orders WHERE id = ? AND user_id = ?',
    [req.params.id, req.user.id],
  );


  if (!order) return res.status(404).json({ error: 'Order not found' });
  res.json(order);
});
```


Return `404` rather than `403` for resources the caller may not see, so the
response does not confirm that the resource exists.


---


## 06 Input Validation


Validate at the boundary — every place untrusted data enters the system: HTTP
handlers, form submissions, webhook payloads, query parameters, file uploads,
and third-party API responses.


Once past the boundary, code can trust its inputs. That is the payoff: internal
functions stay clean because validation happened once, at the edge.


### Validate early, at the entry point


```ts
// ❌ Validation deferred to the point of use
function processOrder(orderId) {
  return db.query('SELECT * FROM orders WHERE id = ?', [orderId]);
}


// ✅ Validated on entry, with a clear contract
function processOrder(orderId: number): Order | null {
  if (!Number.isInteger(orderId) || orderId <= 0) {
    throw new ValidationError('Order ID must be a positive integer');
  }


  const rows = db.query('SELECT * FROM orders WHERE id = ?', [orderId]);
  return rows[0] ?? null;
}
```


### Prefer a schema over hand-rolled checks


At a boundary, declare the shape once and get parsing, validation, and types
from a single source:


```ts
const registrationSchema = z.object({
  email: z.string().email().max(254),
  password: z.string().min(8).regex(/[A-Z]/, 'Must contain an uppercase letter'),
  displayName: z.string().min(1).max(50).trim(),
});


app.post('/api/register', async (req, res) => {
  const parsed = registrationSchema.safeParse(req.body);


  if (!parsed.success) {
    return res.status(400).json({ errors: parsed.error.flatten().fieldErrors });
  }


  // parsed.data is validated and fully typed from here on.
  const user = await userService.register(parsed.data);
  res.status(201).json({ id: user.id });
});
```


### Document non-obvious rules


Encode the reason, especially where a limit comes from outside:


```ts
/**
 * Validates a user email address.
 *
 * Length capped at 254 characters per RFC 5321; addresses longer than that are
 * not deliverable regardless of format.
 *
 * @throws {ValidationError} If the address is missing or malformed.
 */
function validateEmail(email: string): void {
  // ...
}
```


---


## 07 Testing Strategy


**Write the test before the code it covers.** A bug fix without a regression
test is incomplete — that test is the only thing stopping the bug from coming
back.


### Test first, then implement


Every change follows the same loop:


1. **Write a test** that describes the behaviour you want, or reproduces the
   bug you are fixing.
2. **Run it and watch it fail.** Check that it fails for the reason you
   expect — a missing function or a wrong result, not a typo or a broken
   import. A test you have never seen fail may not be testing anything.
3. **Write the least code that makes it pass.**
4. **Refactor** with the test as a safety net, keeping it green.


Tests and implementation are written in that order but **committed together**.
A commit containing only failing tests turns CI red (§03) and breaks
`git bisect` for everyone after you.


Test-first is the default, not a ritual. Where the shape of the code is
genuinely unknown — a spike, a prototype of a third-party integration — explore
freely, then throw the spike away and rebuild it test-first before it merges.


Three layers, each earning its cost.


### Unit tests — pure logic (fast, many)


Target: validators, formatters, calculations, reducers, utilities. No network,
no database, no DOM.


```ts
describe('isValidEmail', () => {
  it('accepts valid addresses', () => {
    expect(isValidEmail('first.last+tag@example.co.uk')).toBe(true);
  });


  it('rejects addresses without @', () => {
    expect(isValidEmail('userexample.com')).toBe(false);
  });


  it('rejects addresses over the RFC 5321 length limit', () => {
    expect(isValidEmail('a'.repeat(300) + '@example.com')).toBe(false);
  });


  it('throws on non-string input', () => {
    expect(() => isValidEmail(null as never)).toThrow();
  });
});
```


Test behaviour, not implementation. A test that breaks when you rename a
private method without changing behaviour is a liability.


### Integration tests — components talking to each other (medium, some)


Target: an API route against a real test database; a form component with its
real validation and store.


Mock only what you cannot run locally — payment providers, email delivery,
third-party APIs. **Do not mock your own database.** Use SQLite or a disposable
container, so schema mistakes actually surface. Mocking the database means your
tests pass and production breaks.


```ts
describe('POST /api/register', () => {
  beforeEach(async () => await resetTestDatabase());


  it('creates a user and queues a welcome email', async () => {
    const res = await request(app)
      .post('/api/register')
      .send({ email: 'new@example.com', password: 'SecurePass123' });


    expect(res.status).toBe(201);
    expect(await db.user.findByEmail('new@example.com')).toBeTruthy();
    expect(emailQueue.pending()).toHaveLength(1);
  });


  it('rejects a duplicate email with 409', async () => {
    await createUser({ email: 'taken@example.com' });


    const res = await request(app)
      .post('/api/register')
      .send({ email: 'taken@example.com', password: 'SecurePass123' });


    expect(res.status).toBe(409);
  });


  it('does not create a user when the password is too weak', async () => {
    const res = await request(app)
      .post('/api/register')
      .send({ email: 'weak@example.com', password: 'short' });


    expect(res.status).toBe(400);
    expect(await db.user.findByEmail('weak@example.com')).toBeNull();
  });
});
```


That last test matters more than it looks: it verifies nothing was
half-written. Check the state, not just the status code.


### End-to-end tests — real browser, real stack (slow, few)


Target: only the journeys that would embarrass you if broken. For most hobby
sites that is three to eight flows — sign up, log in, the core action the site
exists for, and checkout if money is involved.


Resist adding more. E2E tests are the most expensive tests to maintain, and a
bloated suite gets disabled rather than fixed.


```ts
test('a visitor can sign up and reach the dashboard', async ({ page }) => {
  await page.goto('/signup');
  await page.getByLabel('Email').fill(uniqueEmail());
  await page.getByLabel('Password').fill('SecurePass123');
  await page.getByRole('button', { name: 'Create account' }).click();


  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
});
```


**Rules that keep e2e tests from becoming flaky:**


- Select by role, label, or visible text — never by CSS class or DOM position.
  Class-based selectors turn every restyle into a test failure.
- Use the framework's auto-waiting assertions. Never `sleep` or
  `waitForTimeout`; a fixed delay is either too short (flaky) or too long
  (slow), and usually both on different machines.
- Each test creates its own data and does not depend on test order or on
  leftovers from another test.
- Run against a seeded test environment. Never against production.
- **One flaky test that gets ignored destroys trust in the whole suite.** Fix
  it or delete it in the same week you notice it.


### What not to test


Third-party library internals, framework behaviour, trivial getters, exact CSS
values, generated types. Testing these produces churn without safety.


### Coverage


**No coverage percentage gate.** A threshold drives people to write tests for
getters to hit a number, which is worse than not testing at all because it
looks like safety.


The bar instead:


- Every critical user journey has an e2e test.
- Every branch of business logic has a unit test.
- Every fixed bug has a regression test.


Use coverage reports as a map of untested areas, not as a target.


---


## 08 Clean, Robust, Modular Code


Code should be simple to read and hard to break. The reader you are writing for
is a contributor who has never seen this file.


### Prefer strong typing


```ts
// ❌ Weak typing — the shape of `data` is anyone's guess
function process(data) {
  return data.map(x => x * 2);
}


// ✅ Strong typing — the signature is the documentation
function doubleAll(numbers: number[]): number[] {
  return numbers.map(num => num * 2);
}
```


Avoid `any`. If you genuinely cannot type something, use `unknown` and narrow
it — that forces the check rather than skipping it.


### Keep functions small and focused


```ts
// ❌ One function, four responsibilities
function getUserAndValidate(id) {
  const user = db.query(`SELECT * FROM users WHERE id = ${id}`);
  if (!user.email) throw new Error('Missing email');
  if (!user.email.includes('@')) throw new Error('Invalid email');
  const posts = db.query(`SELECT * FROM posts WHERE user_id = ${id}`);
  return { user, postCount: posts.length };
}


// ✅ Each function does one thing and is testable alone
function getUser(userId: number): User | null {
  const rows = db.query('SELECT * FROM users WHERE id = ?', [userId]);
  return rows[0] ?? null;
}


function assertUserEmailIsValid(user: User): void {
  if (!user.email) throw new ValidationError('Email is required');
  if (!isValidEmail(user.email)) throw new ValidationError('Invalid email format');
}


function getUserWithPostCount(userId: number): UserWithStats {
  const user = getUser(userId);
  if (!user) throw new UserNotFoundError(userId);


  assertUserEmailIsValid(user);


  const [{ count }] = db.query(
    'SELECT COUNT(*) AS count FROM posts WHERE user_id = ?',
    [userId],
  );


  return { ...user, postCount: count };
}
```


Note that `getUser` returns `User | null`, not `User`. A query returns rows;
pretending otherwise is how `undefined is not an object` reaches production.


### Extract shared knowledge (DRY)


Each piece of *knowledge* should have one unambiguous representation. The unit
is knowledge, not lines of text — two functions that happen to look alike but
change for different reasons should stay separate.


```ts
// ❌ The email rule lives in two places and will drift
function createUser(email: string) {
  if (!email.includes('@')) throw new Error('Invalid email');
  // ...
}


function updateUser(id: number, email: string) {
  if (!email.includes('@')) throw new Error('Invalid email');
  // ...
}


// ✅ One definition of "valid email"
function validateEmail(email: string): void {
  if (!email) throw new ValidationError('Email is required');
  if (!isValidEmail(email)) throw new ValidationError('Invalid email format');
}
```


**Wait for the third occurrence.** Abstracting on the second is how you get a
helper with five boolean parameters that nobody can safely change.


### Prefer composition over inheritance


```ts
// ❌ Deep hierarchy — every level is a constraint on every subclass
class Shape {}
class Polygon extends Shape {}
class Quadrilateral extends Polygon {}
class Rectangle extends Quadrilateral {}


// ✅ Flat, explicit, easy to extend
interface Shape {
  area(): number;
  perimeter(): number;
}


class Rectangle implements Shape {
  constructor(
    private readonly width: number,
    private readonly height: number,
  ) {}


  area(): number { return this.width * this.height; }
  perimeter(): number { return 2 * (this.width + this.height); }
}
```


Inheritance is acceptable for genuinely shared identity (a base entity with
`id` / `createdAt`, or a framework class you must extend). It is not the tool
for sharing utility behaviour — use a function or a module for that. If you
need a third level of inheritance, the design is wrong.


### Minimize parameters


```ts
// ❌ Nine positional parameters — every call site is a puzzle
function createUser(firstName, lastName, email, phone, address, city, state, zip, country) {}


// ✅ Grouped, named, optional where optional
interface CreateUserInput {
  firstName: string;
  lastName: string;
  email: string;
  phone?: string;
  address?: {
    street: string;
    city: string;
    state: string;
    zip: string;
    country: string;
  };
}


function createUser(input: CreateUserInput): Promise<User> {
  // ...
}
```


### Guard against unexpected state


Defensive checks belong where bad input can actually arrive — boundaries and
public APIs. Do not re-validate the same value in five internal helpers; that
is noise, and it hides where the real contract is.


```ts
function calculateAverage(numbers: number[]): number {
  if (numbers.length === 0) {
    throw new ValidationError('Cannot average an empty array');
  }


  return numbers.reduce((sum, n) => sum + n, 0) / numbers.length;
}
```


---


## 09 Assert Invariants


An invariant is a condition that must hold if the program is correct. Violating
one means **the code has a bug**, not that the user did something wrong.


**Do not use `console.assert`.** It does not throw and does not halt execution —
it prints a message and carries on, which is the opposite of failing fast, and
it is stripped or silenced in some production builds. Use a real helper:


```ts
// src/shared/invariant.ts
export function invariant(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(`Invariant violated: ${message}`);
  }
}
```


The `asserts condition` return type also narrows types for the compiler, so
guards double as type refinement.


### Invariants versus validation


This distinction matters, and getting it backwards produces either crashes on
normal user behaviour or silent data corruption:


| | Invariant | Validation |
|---|---|---|
| Cause | Programmer error | User or network input |
| Example | "cart total is never negative" | "insufficient funds" |
| Response | Crash loudly, log, alert | Friendly message to the user |
| Tool | `invariant()` | Schema check, `ValidationError` |


```ts
// ❌ Wrong: insufficient funds is a normal user condition, not a bug
function withdraw(account: Account, amount: number): void {
  invariant(amount <= account.balance, 'Insufficient funds');
  account.balance -= amount;
}


// ✅ Right: expected conditions are validated, impossible states are asserted
function withdraw(account: Account, amount: number): Result<void, string> {
  if (amount <= 0) {
    return { ok: false, error: 'Withdrawal amount must be positive' };
  }


  if (amount > account.balance) {
    return { ok: false, error: 'Insufficient funds for this withdrawal' };
  }


  account.balance -= amount;


  // A negative balance here means the arithmetic or a concurrent write is
  // broken. There is no safe way to continue.
  invariant(account.balance >= 0, 'Balance went negative after withdrawal');


  return { ok: true, value: undefined };
}
```


Use invariants to close impossible branches — an unreachable `switch` default,
a value the type system cannot prove is present:


```ts
function labelFor(status: OrderStatus): string {
  switch (status) {
    case 'pending':   return 'Awaiting payment';
    case 'shipped':   return 'On its way';
    case 'delivered': return 'Delivered';
    default:
      // Reached only if a new status was added without updating this switch.
      invariant(false, `Unhandled order status: ${status}`);
  }
}
```


---


## 10 Error Handling


Ask one question first: **is this an exceptional failure, or an expected one?**


Expected failures — invalid input, a missing optional record, a declined
card — are part of normal operation and belong in the return type. Exceptional
failures — a database that is down, a broken invariant — should propagate and
alert someone.


### Throw for exceptional conditions


```ts
function connectToDatabase(connectionString: string): Connection {
  if (!connectionString) {
    throw new Error('DATABASE_URL is required');
  }


  try {
    return db.connect(connectionString);
  } catch (error) {
    // A database outage is not something this layer can meaningfully handle.
    throw new DatabaseConnectionError('Failed to connect to database', { cause: error });
  }
}
```


### Return a value for expected conditions


```ts
function fetchUserPreferences(userId: number): UserPreferences {
  const rows = db.query('SELECT * FROM user_preferences WHERE user_id = ?', [userId]);


  // Users who have never opened settings have no row yet. That is normal.
  return rows[0] ?? getDefaultPreferences();
}
```


### Separate the user-facing message from the diagnostic detail


```ts
async function updateUserEmail(
  userId: number,
  newEmail: string,
): Promise<Result<void, string>> {
  if (!isValidEmail(newEmail)) {
    return { ok: false, error: 'Please provide a valid email address' };
  }


  try {
    await db.query('UPDATE users SET email = ? WHERE id = ?', [newEmail, userId]);
    return { ok: true, value: undefined };
  } catch (error) {
    logger.error('Failed to update email', { userId, error });
    return { ok: false, error: 'Unable to update email. Please try again later.' };
  }
}
```


### Rules


- **Never swallow an error silently.** An empty `catch {}` turns a five-minute
  bug into a two-day bug.
- **Never leak internals to users.** Stack traces, SQL, and file paths in an
  error response are both confusing and an information disclosure.
- **Preserve the cause** when wrapping: `new Error(msg, { cause: error })`.
- **Use typed error classes** for cases callers need to distinguish
  (`UserNotFoundError`, `ValidationError`), not string matching on messages.
- **Show users a next action**, not just a failure: "Try again in a minute" or
  "Check your card details", never "Error 500".


### Handle failure in the UI too


Every asynchronous UI state has three outcomes, and all three need a rendered
state: loading, success, and error. A spinner that spins forever on failure is
the most common bug in hobby web apps.


---


## 11 Code Comments: Explain the "Why"


Comments explain reasoning, constraints, and intent. Naming and structure
explain the rest. A comment restating the code is worse than nothing, because
it drifts out of date and then actively lies.


```ts
// ❌ Restates the code
function calculateDiscount(price, quantity) {
  // Multiply price by quantity
  let total = price * quantity;


  // If quantity is greater than 10, discount is 10%
  if (quantity > 10) total = total * 0.9;


  return total;
}


// ✅ Explains the reasoning a reader cannot recover from the code
function calculateDiscount(price: number, quantity: number): number {
  let total = price * quantity;


  // Bulk discount kicks in at 10 units to match the wholesale pricing tier
  // our supplier gives us, so the margin stays positive below that threshold.
  if (quantity > BULK_DISCOUNT_THRESHOLD) {
    total *= 1 - BULK_DISCOUNT_RATE;
  }


  return total;
}
```


### Document workarounds, constraints, and expiry conditions


```ts
// The upstream API returns 200 with an empty body on rate limit instead of 429,
// so an empty response has to be treated as retryable. Remove once they fix it.
// Tracked in issue #142.
if (response.status === 200 && !response.body) {
  return retryWithBackoff();
}
```


### Explain unusual approaches


```ts
// Floyd's cycle detection rather than a Set: this runs in O(1) space, which
// matters because the input array can be tens of millions of elements.
// https://en.wikipedia.org/wiki/Cycle_detection
function findDuplicate(arr: number[]): number {
  // ...
}
```


### Write full sentences


Comments are prose. Capitalize, punctuate, and write for someone who does not
already know what you meant.


### Never explain the change, only the code


Comments describe the code as it is now. Why you changed it belongs in the
commit message, where it is attached to the diff and does not rot. Do not leave
`// changed this to fix the bug` or `// was previously using a Map` in source.


---


## 12 API and Module Design


Modules should be deep — a simple interface hiding real complexity. A module
whose interface is as complicated as its implementation has not earned its
existence.


### Hide implementation details


```ts
// ❌ Leaks the internal data structure; callers now depend on it being a Map
class UserManager {
  users: Map<number, User> = new Map();


  getUser(id: number): User {
    return this.users.get(id);   // may be undefined despite the signature
  }
}


// ✅ Interface says what it does; storage is free to change
class UserManager {
  #users = new Map<number, User>();


  getUser(id: number): User {
    const user = this.#users.get(id);
    if (!user) throw new UserNotFoundError(id);
    return user;
  }


  findUser(id: number): User | null {
    return this.#users.get(id) ?? null;
  }
}
```


Two methods, because "get me the user, it must exist" and "check whether the
user exists" are different questions and should not be answered by the same
ambiguous return value.


### Minimize dependencies


```ts
// ❌ Every caller must assemble seven collaborators
function createUserProfile(
  user, preferences, settings, cache, logger, emailService, analyticsService,
) {}


// ✅ Pass only what the function actually needs
interface UserProfileContext {
  preferences: UserPreferences;
  locale: string;
}


function createUserProfile(user: User, context: UserProfileContext): UserProfile {
  // ...
}
```


If a function needs seven collaborators, it is doing seven things.


### Keep interfaces consistent


```ts
interface Repository<T> {
  create(item: Omit<T, 'id'>): Promise<T>;
  findById(id: string): Promise<T | null>;
  update(id: string, changes: Partial<T>): Promise<T>;
  delete(id: string): Promise<void>;
}
```


Once contributors learn one repository, they know them all. Consistency is
worth more than each module being individually optimal.


---


## 13 Code Organization


Group by feature, not by technical layer. `users/` beats `controllers/`,
`services/`, `models/` — a change to user handling touches one directory
instead of five.


```
src/
├── users/
│   ├── user.model.ts        # Types and data shapes
│   ├── user.repository.ts   # Database access
│   ├── user.service.ts      # Business logic
│   ├── user.validator.ts    # Validation schemas
│   ├── user.routes.ts       # HTTP handlers
│   └── user.test.ts         # Co-located tests
├── shared/
│   ├── validators/
│   ├── utils/
│   └── types/
├── config.ts                # The only reader of process.env
└── e2e/                     # Browser tests, separate from unit tests
```


- **Co-locate tests** with the code they test, so moving a feature moves its
  tests and deleting a feature deletes them.
- **Dependencies point inward.** Routes depend on services, services on
  repositories. Never the reverse — a repository that imports a route handler
  makes the whole thing untestable.
- **`shared/` is for genuinely shared code**, not a dumping ground. If only one
  feature uses it, it belongs in that feature.


---


## 14 Web Performance


Measure before optimizing. For websites, these dominate — and none of them is
the clever algorithm you were about to write:


- **N+1 queries are the number one cause of slow pages.** Fetch related data in
  one query or batch it. Log query counts per request in development so a new
  N+1 is visible immediately.
- **Images.** Correct dimensions, modern formats, lazy-load below the fold, and
  always set `width`/`height` to prevent layout shift. An unoptimized hero
  image outweighs every JavaScript optimization you will ever make.
- **Ship less JavaScript.** Check bundle size before adding a dependency — a
  date-formatting library is rarely worth 70 KB. Code-split by route.
- **Index every column you filter or sort on.** A missing index is invisible at
  100 rows and fatal at 100,000.
- **Cache at the edge** whatever does not vary per user.


### Filter at the source


```ts
// ❌ Pulls every row into memory, then throws most of it away
function getActiveUserEmails(): string[] {
  const allUsers = db.query('SELECT * FROM users');
  return allUsers.filter(u => u.isActive).map(u => u.email);
}


// ✅ Let the database do what it is good at
function getActiveUserEmails(): string[] {
  return db.query('SELECT email FROM users WHERE is_active = true');
}
```


### Parallelize independent work


```ts
// ❌ Three sequential round trips
async function getUserWithStats(userId: number): Promise<UserStats> {
  const user = await getUser(userId);
  const posts = await getPostCount(userId);
  const followers = await getFollowerCount(userId);
  return { user, posts, followers };
}


// ✅ One round trip's worth of latency
async function getUserWithStats(userId: number): Promise<UserStats> {
  const [user, posts, followers] = await Promise.all([
    getUser(userId),
    getPostCount(userId),
    getFollowerCount(userId),
  ]);


  return { user, posts, followers };
}
```


### Choose data structures for the actual N


An array `includes` on three roles is fine and clearer than a `Set`. Reach for
a `Set` or `Map` when the collection is large, or when the lookup happens
inside a loop:


```ts
// ✅ O(1) lookups matter here — this runs once per row, for thousands of rows
const allowedIds = new Set(permittedUserIds);
const visible = rows.filter(row => allowedIds.has(row.userId));
```


Use Lighthouse or your framework's bundle analyzer to find the real bottleneck.
It is almost never the code you suspected.


---


## 15 Accessibility and Semantic HTML


Accessibility is not a later phase. Retrofitting costs several times more than
building it in, and it directly improves keyboard usability and SEO.


- **Use the correct element.** A thing that navigates is an `<a>`; a thing that
  acts is a `<button>`. A `<div onClick>` is invisible to keyboards and screen
  readers and gets no focus, no Enter key, and no announcement.
- **Every form control has an associated `<label>`.** Placeholder text is not a
  label — it disappears when typing starts.
- **Every meaningful image has `alt` text**; decorative images get `alt=""`.
- **Everything reachable by mouse is reachable by keyboard**, with a visible
  focus indicator. Never remove focus outlines without providing a replacement.
- **Text contrast at least 4.5:1** against its background.
- **One `<h1>` per page**; heading levels descend without skipping.
- **Errors appear as text next to the field**, not communicated by colour
  alone.


```html
<!-- ❌ Invisible to assistive tech and to the keyboard -->
<div class="btn" onclick="submitForm()">Submit</div>
<input placeholder="Email" />


<!-- ✅ Works for everyone, with less code -->
<button type="submit">Submit</button>


<label for="email">Email</label>
<input id="email" type="email" name="email" required
       aria-describedby="email-error" />
<p id="email-error" role="alert">Please enter a valid email address.</p>
```


**Check before shipping any UI change:** tab through it without touching the
mouse. Most accessibility defects surface in that thirty seconds.


---


## 16 Logging and Monitoring


Log enough to answer "what happened?" after the fact, without creating a
liability.


### Never log sensitive data


Logs get stored, shipped to third parties, and read by anyone with dashboard
access. **Never log** passwords, tokens, session cookies, API keys, full credit
card numbers, or full email addresses. Reference users by `userId`.


```ts
// ❌ Puts personal data in log storage indefinitely
logger.warn('Failed login attempt', { email, password, ipAddress });


// ✅ Debuggable without storing personal data
logger.warn('Failed login attempt', {
  userId: user?.id ?? null,
  emailDomain: email.split('@')[1],
  failureReason: 'invalid_password',
  attemptNumber: 3,
});
```


If you must correlate by an identifier you cannot store, log a salted hash.


### Use levels meaningfully


```ts
logger.debug('Query executed', { query: 'users.findById', durationMs: 42 });
logger.info('User registered', { userId: user.id });
logger.warn('Cache miss for user profile', { userId, reason: 'ttl_expired' });
logger.error('Database connection failed', { attempt: 3, error });
```


- `debug` — development detail, off in production.
- `info` — significant business events worth counting.
- `warn` — something recoverable that a human may want to know about.
- `error` — something failed and needs attention. If nobody would act on it, it
  is not an error.


Include structured context, never string-interpolated blobs. Structured fields
are searchable; concatenated strings are not.


### Track what would tell you the site is broken


```ts
const startedAt = performance.now();
const results = await expensiveQuery();
const durationMs = performance.now() - startedAt;


if (durationMs > SLOW_QUERY_THRESHOLD_MS) {
  logger.warn('Slow query detected', {
    query: 'user_profile',
    durationMs,
    thresholdMs: SLOW_QUERY_THRESHOLD_MS,
  });
}
```


At hobby scale that is: an uptime check, error reporting with alerts, and slow
query logging. That is enough. Do not build a metrics pipeline for a site with
200 users.


### Log security events


Authentication failures, authorization denials, and rate-limit trips — with
`userId` and outcome, never credentials.


---


## 17 Definition of Done


### Must — do not merge without these


- [ ] **No secrets committed**, and nothing sensitive in logs
- [ ] **Inputs validated at every boundary**, on the server
- [ ] **Every handler authorizes the caller** for the specific resource
- [ ] **No string-concatenated SQL, HTML, or shell commands**
- [ ] **Errors handled** — nothing silently swallowed, no internals leaked to
      users
- [ ] **Tests written first and passing** — unit tests for logic, integration
      tests for the route, e2e if a critical journey changed
- [ ] **Bug fixes include a regression test** that was seen to fail before the
      fix
- [ ] **CI green** — format, lint, typecheck, tests
- [ ] **Keyboard accessible**, labelled, with visible focus (UI changes)
- [ ] **README / `.env.example` updated** if setup or configuration changed


### Should — justify skipping these


- [ ] Functions are small and single-purpose
- [ ] Types are precise; no stray `any`
- [ ] Invariants asserted where an impossible state would otherwise pass
      silently
- [ ] Comments explain *why*; none restate the code
- [ ] No duplicated knowledge (at the third occurrence, extract)
- [ ] Config over hardcoded values
- [ ] Logging sufficient to debug this code in production
- [ ] No obvious N+1 query or unoptimized image
- [ ] Commits are small and coherent; pull request explains what and why


---


## 18 Worked Example: User Registration


How the guide applies to one real feature, split into reviewable changes. Each
change starts with tests, watched failing, then adds the code that makes them
pass. Tests and code land in the same commit (§07).


### Change 1 — validation schema


**Tests first.** Run them and they fail, because `registrationSchema` does not
exist yet.


```ts
// src/users/user.validator.test.ts
describe('registrationSchema', () => {
  it('accepts a valid registration', () => {
    const result = registrationSchema.safeParse({
      email: 'user@example.com',
      password: 'SecurePass123',
    });


    expect(result.success).toBe(true);
  });


  it('normalizes email casing and whitespace', () => {
    const result = registrationSchema.parse({
      email: '  User@Example.COM ',
      password: 'SecurePass123',
    });


    expect(result.email).toBe('user@example.com');
  });


  it('rejects a password with no uppercase letter', () => {
    const result = registrationSchema.safeParse({
      email: 'user@example.com',
      password: 'securepass123',
    });


    expect(result.success).toBe(false);
  });


  it('rejects an email over the RFC 5321 length limit', () => {
    const result = registrationSchema.safeParse({
      email: 'a'.repeat(250) + '@example.com',
      password: 'SecurePass123',
    });


    expect(result.success).toBe(false);
  });
});
```


**Then the implementation** that makes them pass:


```ts
// src/users/user.validator.ts
import { z } from 'zod';


// Email capped at 254 characters per RFC 5321.
export const registrationSchema = z.object({
  email: z.string().trim().toLowerCase().email().max(254),
  password: z
    .string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Password must contain an uppercase letter')
    .regex(/[0-9]/, 'Password must contain a number'),
});


export type RegistrationInput = z.infer<typeof registrationSchema>;
```


### Change 2 — repository layer


**Tests first,** against the real test database. The duplicate-email test
also proves the unique index on `users.email` exists; if the migration is
missing, it fails here rather than in production.


```ts
// src/users/user.repository.test.ts
describe('UserRepository', () => {
  const repo = new UserRepository();


  beforeEach(async () => await resetTestDatabase());


  it('creates a user that can then be found by email', async () => {
    const created = await repo.create('new@example.com', 'hashed-value');


    const found = await repo.findByEmail('new@example.com');
    expect(found?.id).toBe(created.id);
  });


  it('returns null for an unknown email', async () => {
    expect(await repo.findByEmail('nobody@example.com')).toBeNull();
  });


  it('throws UserAlreadyExistsError on a duplicate email', async () => {
    await repo.create('taken@example.com', 'hashed-value');


    await expect(repo.create('taken@example.com', 'other-hash'))
      .rejects.toBeInstanceOf(UserAlreadyExistsError);
  });
});
```


**Then the implementation:**


```ts
// src/users/user.repository.ts
export class UserRepository {
  async findByEmail(email: string): Promise<User | null> {
    const rows = await db.query('SELECT * FROM users WHERE email = ?', [email]);
    return rows[0] ?? null;
  }


  async create(email: string, passwordHash: string): Promise<User> {
    try {
      const result = await db.query(
        'INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)',
        [email, passwordHash, new Date()],
      );


      return { id: result.insertId, email, createdAt: new Date() };
    } catch (error) {
      // Relies on the unique index on users.email rather than a check-then-insert,
      // which would race between two concurrent signups.
      if (isUniqueViolation(error)) {
        throw new UserAlreadyExistsError(email);
      }
      throw error;
    }
  }
}
```


### Change 3 — service layer


**Tests first.** The repository and database are real. Only the logger is
stubbed, because logging is not the behaviour under test.


```ts
// src/users/user.service.test.ts
describe('UserService.register', () => {
  const users = new UserRepository();
  const service = new UserService(
    users,
    new PasswordService(),
    { info: vi.fn(), warn: vi.fn(), error: vi.fn() } as unknown as Logger,
  );


  beforeEach(async () => await resetTestDatabase());


  it('returns the new user on success', async () => {
    const result = await service.register({
      email: 'new@example.com',
      password: 'SecurePass123',
    });


    expect(result.ok).toBe(true);
  });


  it('returns EMAIL_IN_USE for a duplicate email', async () => {
    const input = { email: 'taken@example.com', password: 'SecurePass123' };
    await service.register(input);


    const result = await service.register(input);
    expect(result).toEqual({ ok: false, error: 'EMAIL_IN_USE' });
  });


  it('never stores the password in plain text', async () => {
    await service.register({ email: 'hash@example.com', password: 'SecurePass123' });


    const user = await users.findByEmail('hash@example.com');
    expect(user!.passwordHash).not.toContain('SecurePass123');
  });
});
```


**Then the implementation:**


```ts
// src/users/user.service.ts
export class UserService {
  constructor(
    private readonly users: UserRepository,
    private readonly passwords: PasswordService,
    private readonly logger: Logger,
  ) {}


  async register(input: RegistrationInput): Promise<Result<User, RegistrationError>> {
    const passwordHash = await this.passwords.hash(input.password);


    try {
      const user = await this.users.create(input.email, passwordHash);
      this.logger.info('User registered', { userId: user.id });
      return { ok: true, value: user };
    } catch (error) {
      if (error instanceof UserAlreadyExistsError) {
        this.logger.info('Registration rejected: email in use', {
          emailDomain: input.email.split('@')[1],
        });
        return { ok: false, error: 'EMAIL_IN_USE' };
      }


      this.logger.error('Registration failed', { error });
      return { ok: false, error: 'INTERNAL' };
    }
  }
}
```


### Change 4 — HTTP route


**Tests first:**


```ts
// src/users/user.routes.test.ts
describe('POST /register', () => {
  beforeEach(async () => await resetTestDatabase());


  it('creates a user and starts a session', async () => {
    const res = await request(app)
      .post('/register')
      .send({ email: 'new@example.com', password: 'SecurePass123' });


    expect(res.status).toBe(201);
    expect(res.headers['set-cookie']).toBeDefined();
    expect(await userRepository.findByEmail('new@example.com')).toBeTruthy();
  });


  it('rejects a duplicate email without revealing that it exists', async () => {
    await createUser({ email: 'taken@example.com' });


    const res = await request(app)
      .post('/register')
      .send({ email: 'taken@example.com', password: 'SecurePass123' });


    expect(res.status).toBe(409);
    expect(res.body.error).not.toContain('taken@example.com');
  });


  it('rejects a weak password without creating a user', async () => {
    const res = await request(app)
      .post('/register')
      .send({ email: 'weak@example.com', password: 'short' });


    expect(res.status).toBe(400);
    expect(await userRepository.findByEmail('weak@example.com')).toBeNull();
  });
});
```


**Then the implementation:**


```ts
// src/users/user.routes.ts
router.post('/register', rateLimit({ max: 5, windowMs: 60_000 }), async (req, res) => {
  const parsed = registrationSchema.safeParse(req.body);


  if (!parsed.success) {
    return res.status(400).json({ errors: parsed.error.flatten().fieldErrors });
  }


  const result = await userService.register(parsed.data);


  if (!result.ok) {
    // Deliberately vague, so the endpoint cannot be used to enumerate
    // which email addresses have accounts.
    return result.error === 'EMAIL_IN_USE'
      ? res.status(409).json({ error: 'Could not create account with those details' })
      : res.status(500).json({ error: 'Something went wrong. Please try again.' });
  }


  await createSession(res, result.value);
  res.status(201).json({ id: result.value.id });
});
```


### Change 5 — signup page


**E2E tests first.** They describe the journey before the page exists:


```ts
// e2e/registration.spec.ts
test('a visitor can register and land on the dashboard', async ({ page }) => {
  await page.goto('/signup');
  await page.getByLabel('Email').fill(uniqueEmail());
  await page.getByLabel('Password').fill('SecurePass123');
  await page.getByRole('button', { name: 'Create account' }).click();


  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
});


test('the signup form reports a weak password inline', async ({ page }) => {
  await page.goto('/signup');
  await page.getByLabel('Email').fill(uniqueEmail());
  await page.getByLabel('Password').fill('short');
  await page.getByRole('button', { name: 'Create account' }).click();


  await expect(page.getByRole('alert')).toContainText('at least 8 characters');
});
```


**Then build the form** until both pass. It is omitted here; follow §15 for
labels, keyboard access, and inline error text. Because the tests select by
label and role, a form that passes them is already most of the way to
accessible.


---


## Summary


Ordered by how much each protects the project:


1. **Setup works** — one command, honest README, `.env.example` committed
2. **Small, coherent changes** — reviewable, revertable, explained
3. **Automation enforces the rules** — lint, typecheck, CI on every pull request
4. **No secrets anywhere** — one config module, validated at startup
5. **Security baseline holds** — parameterized queries, server-side authorization,
   proper password hashing
6. **Inputs validated at the boundary** — schema at the edge, trust inside
7. **Tests first, matched to risk** — written before the code, many unit, some
   integration, few e2e, always a regression test
8. **Code is clean and modular** — small functions, precise types, composition
   over inheritance
9. **Invariants assert real bugs** — `invariant()`, never `console.assert`
10. **Errors handled honestly** — nothing swallowed, nothing leaked
11. **Comments explain why** — the "what" is the code's job
12. **APIs are deep and narrow** — simple interfaces over real complexity
13. **Performance where it is measured** — queries, images, bundle size
14. **Accessible by default** — semantic HTML, labels, keyboard support
15. **Logs are useful and safe** — structured, levelled, free of personal data


The point of all of it is that the next contributor — six months from now,
possibly you, having forgotten everything — can make a change with confidence.