(() => {
    if (window.__proofPathLightboxInitialized) {
        return;
    }

    window.__proofPathLightboxInitialized = true;

    const triggerSelector = ".proof-path-lightbox-trigger[data-proof-path-lightbox]";
    let overlayElements = null;
    let activeTrigger = null;
    let schemeObserver = null;

    function currentScheme() {
        return document.body?.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "light";
    }

    function getFocusableElements(container) {
        return Array.from(
            container.querySelectorAll("button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])")
        ).filter((element) => !element.hasAttribute("disabled"));
    }

    function ensureOverlay() {
        if (overlayElements && document.body?.contains(overlayElements.root)) {
            return overlayElements;
        }

        const root = document.createElement("div");
        root.className = "proof-path-lightbox";
        root.hidden = true;
        root.innerHTML = `
            <div class="proof-path-lightbox__backdrop"></div>
            <div
                class="proof-path-lightbox__dialog"
                role="dialog"
                aria-modal="true"
                aria-labelledby="proof-path-lightbox-title"
                aria-describedby="proof-path-lightbox-caption"
                tabindex="-1"
            >
                <div class="proof-path-lightbox__header">
                    <p class="proof-path-lightbox__title" id="proof-path-lightbox-title">Enlarged plot</p>
                    <button type="button" class="proof-path-lightbox__close" aria-label="Close enlarged plot">Close</button>
                </div>
                <figure class="proof-path-lightbox__figure">
                    <div class="proof-path-lightbox__image-shell">
                        <img class="proof-path-lightbox__image" alt="" />
                    </div>
                    <figcaption class="proof-path-lightbox__caption" id="proof-path-lightbox-caption"></figcaption>
                </figure>
            </div>
        `;

        document.body.appendChild(root);

        const dialog = root.querySelector(".proof-path-lightbox__dialog");
        const closeButton = root.querySelector(".proof-path-lightbox__close");
        const backdrop = root.querySelector(".proof-path-lightbox__backdrop");

        backdrop.addEventListener("click", () => closeLightbox());
        closeButton.addEventListener("click", () => closeLightbox());
        dialog.addEventListener("click", (event) => event.stopPropagation());

        overlayElements = {
            root,
            dialog,
            title: root.querySelector(".proof-path-lightbox__title"),
            image: root.querySelector(".proof-path-lightbox__image"),
            caption: root.querySelector(".proof-path-lightbox__caption"),
            closeButton
        };

        return overlayElements;
    }

    function getTriggerImageSource(trigger) {
        const lightSrc = trigger.dataset.lightSrc || trigger.getAttribute("href") || "";
        const darkSrc = trigger.dataset.darkSrc || lightSrc;
        return currentScheme() === "dark" ? darkSrc : lightSrc;
    }

    function syncOverlayContent(trigger) {
        const elements = ensureOverlay();
        const figure = trigger.closest("figure");
        const figcaption = figure?.querySelector("figcaption");

        elements.title.textContent = trigger.dataset.lightboxTitle || "Enlarged plot";
        elements.image.src = getTriggerImageSource(trigger);
        elements.image.alt = trigger.dataset.alt || "";
        elements.caption.innerHTML = figcaption ? figcaption.innerHTML : "";
    }

    function openLightbox(trigger) {
        const elements = ensureOverlay();
        activeTrigger = trigger;
        syncOverlayContent(trigger);
        elements.root.hidden = false;
        document.body.classList.add("proof-path-lightbox-open");
        elements.closeButton.focus();
    }

    function closeLightbox({ restoreFocus = true } = {}) {
        if (!overlayElements || overlayElements.root.hidden) {
            activeTrigger = null;
            return;
        }

        overlayElements.root.hidden = true;
        overlayElements.image.removeAttribute("src");
        overlayElements.image.alt = "";
        overlayElements.caption.innerHTML = "";
        document.body.classList.remove("proof-path-lightbox-open");

        const triggerToRestore = activeTrigger;
        activeTrigger = null;

        if (restoreFocus && triggerToRestore && document.contains(triggerToRestore)) {
            triggerToRestore.focus();
        }
    }

    function bindTrigger(trigger) {
        if (trigger.dataset.proofPathLightboxBound === "true") {
            return;
        }

        trigger.dataset.proofPathLightboxBound = "true";

        trigger.addEventListener("click", (event) => {
            if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
                return;
            }

            event.preventDefault();
            openLightbox(trigger);
        });

        trigger.addEventListener("keydown", (event) => {
            if (event.key !== " " && event.key !== "Spacebar" && event.key !== "Enter") {
                return;
            }

            event.preventDefault();
            openLightbox(trigger);
        });
    }

    function bindCurrentPageTriggers() {
        closeLightbox({ restoreFocus: false });
        document.querySelectorAll(triggerSelector).forEach(bindTrigger);
    }

    function handleDocumentKeydown(event) {
        if (!overlayElements || overlayElements.root.hidden) {
            return;
        }

        if (event.key === "Escape") {
            event.preventDefault();
            closeLightbox();
            return;
        }

        if (event.key !== "Tab") {
            return;
        }

        const focusable = getFocusableElements(overlayElements.dialog);
        if (focusable.length === 0) {
            event.preventDefault();
            overlayElements.dialog.focus();
            return;
        }

        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        const activeElement = document.activeElement;

        if (event.shiftKey) {
            if (activeElement === first || activeElement === overlayElements.dialog) {
                event.preventDefault();
                last.focus();
            }
            return;
        }

        if (activeElement === last) {
            event.preventDefault();
            first.focus();
        }
    }

    function ensureSchemeObserver() {
        if (schemeObserver || !document.body) {
            return;
        }

        schemeObserver = new MutationObserver((mutations) => {
            if (!activeTrigger || !overlayElements || overlayElements.root.hidden) {
                return;
            }

            const schemeChanged = mutations.some((mutation) => mutation.attributeName === "data-md-color-scheme");
            if (schemeChanged) {
                syncOverlayContent(activeTrigger);
            }
        });

        schemeObserver.observe(document.body, {
            attributes: true,
            attributeFilter: ["data-md-color-scheme"]
        });
    }

    function subscribeToInstantNavigation() {
        const documentStream = window.document$;
        if (!documentStream || typeof documentStream.subscribe !== "function") {
            return false;
        }

        documentStream.subscribe(() => {
            bindCurrentPageTriggers();
        });

        return true;
    }

    function bootstrapProofPathLightbox() {
        document.addEventListener("keydown", handleDocumentKeydown);
        ensureSchemeObserver();
        bindCurrentPageTriggers();

        if (!subscribeToInstantNavigation()) {
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
        window.addEventListener("DOMContentLoaded", bootstrapProofPathLightbox, { once: true });
    } else {
        bootstrapProofPathLightbox();
    }
})();
