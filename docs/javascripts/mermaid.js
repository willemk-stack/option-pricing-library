// Mermaid initialization for MkDocs Material.
// This is deliberately tiny: we just auto-render any `mermaid` code fences.

(function initMermaid() {
    // Mermaid is loaded from CDN via mkdocs.yml (extra_javascript).
    if (typeof mermaid === "undefined") return;

    mermaid.initialize({
        startOnLoad: true,
    });
})();
