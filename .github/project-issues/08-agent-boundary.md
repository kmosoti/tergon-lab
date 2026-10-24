# component: validate App identities before enabling scoped agent merges

Register/install the App and keep its key in a trusted broker. Run the acceptance
tests in docs/wiki/GitHub-App-setup.md in a disposable repository. Contributor
tokens must fail direct-main writes and check forgery; stale head/base approvals
must fail; protected changes must retain owner approval. Record permission-denial
receipts. Only then implement a default-branch scope classifier and trusted merger
and propose enabling policy/agent-scopes.json. A flag or label alone is insufficient.
