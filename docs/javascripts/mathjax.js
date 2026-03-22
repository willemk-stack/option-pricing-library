window.MathJax = {
    tex: {
        inlineMath: [["$", "$"], ["\\(", "\\)"]],
        displayMath: [["$$", "$$"], ["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true
    },
    svg: {
        fontCache: "global"
    },
    options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
    }
};

let isTypesetting = false;
let pendingTypeset = false;

function hasMathPlaceholders() {
    return document.querySelector(".arithmatex") !== null;
}

function typesetMath() {
    const mathJax = window.MathJax;
    if (!mathJax?.startup?.promise || !hasMathPlaceholders()) {
        return Promise.resolve(false);
    }

    return mathJax.startup.promise
        .then(() => {
            if (mathJax.startup.output?.clearCache) {
                mathJax.startup.output.clearCache();
            }
            mathJax.typesetClear();
            mathJax.texReset();
            return mathJax.typesetPromise().then(() => true);
        })
        .catch((error) => {
            console.error("MathJax typeset failed", error);
            return false;
        });
}

function queueTypeset() {
    if (!hasMathPlaceholders()) {
        return;
    }

    if (isTypesetting) {
        pendingTypeset = true;
        return;
    }

    isTypesetting = true;
    window.requestAnimationFrame(() => {
        typesetMath().finally(() => {
            isTypesetting = false;
            if (pendingTypeset) {
                pendingTypeset = false;
                queueTypeset();
            }
        });
    });
}

function subscribeToInstantNavigation() {
    const documentStream = window.document$;
    if (!documentStream || typeof documentStream.subscribe !== "function") {
        return false;
    }

    documentStream.subscribe(() => {
        queueTypeset();
    });

    return true;
}

function observeMathContent() {
    if (!document.body) {
        return;
    }

    const observer = new MutationObserver((mutations) => {
        const shouldTypeset = mutations.some((mutation) =>
            Array.from(mutation.addedNodes).some((node) => {
                if (node.nodeType !== Node.ELEMENT_NODE) {
                    return false;
                }

                return (
                    node.matches?.(".arithmatex, [data-md-component='content'], article, main") ||
                    node.querySelector?.(".arithmatex")
                );
            })
        );

        if (shouldTypeset) {
            queueTypeset();
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });
}

function bootstrapMathJax() {
    observeMathContent();
    queueTypeset();

    if (!subscribeToInstantNavigation()) {
        if (subscribeToInstantNavigation()) {
            return;
        }

        const retryId = window.setInterval(() => {
            if (subscribeToInstantNavigation()) {
                window.clearInterval(retryId);
            }
        }, 50);

        window.setTimeout(() => {
            window.clearInterval(retryId);
        }, 5000);
    }
}

if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", bootstrapMathJax, { once: true });
} else {
    bootstrapMathJax();
}
