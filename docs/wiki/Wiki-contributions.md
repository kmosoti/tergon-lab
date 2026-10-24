# Wiki contributions

Edit docs/wiki in a normal branch and PR. These files are canonical. Keep one page
per topic, link to numerical contracts and evidence, and distinguish current
behavior from proposed work. CI checks local links and basic prose rules; review
still checks whether claims are supported.

The native GitHub Wiki is a separate Git repository. GitHub requires an initial
page before it can be cloned. The owner must open the Wiki tab and create a Home
page containing exactly `# Tergon Lab` to initialize it. Do not put credentials or
unreviewed research claims there.

After the foundation PR is merged, use the `Publish reviewed wiki` workflow from
main. It runs `cargo xtask wiki-sync` on a checkout of the wiki, verifies the
previous published file hashes, and replaces only the managed pages with the
reviewed docs/wiki source. Every page gets the source commit SHA in its footer.
The first sync accepts only the minimal Home initialization above.

If anyone edits the published pages directly, adds an unmanaged page, or removes
a managed page, sync fails with drift. Reconcile the edit into a PR and restore
the previously published wiki state before retrying. Do not automatically discard
someone's web edits. A deleted sync manifest is also drift, not permission to reset.

To inspect or stage locally from a clean, current main checkout:

```bash
git clone https://github.com/kmosoti/tergon-lab.wiki.git scratch/wiki
cargo xtask wiki-sync scratch/wiki
git -C scratch/wiki diff
```

The command stages files on disk; the workflow performs the commit and push.
The Wiki is a convenience mirror. Source review and numerical evidence remain
in the main repository even when the Wiki has not yet been initialized.
