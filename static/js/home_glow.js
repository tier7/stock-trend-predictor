const homeMain = document.querySelector(".home-page main");
const hasFinePointer = window.matchMedia("(pointer: fine)").matches;
const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
).matches;

if (homeMain && hasFinePointer && !prefersReducedMotion) {
    let pointerX = 0;
    let pointerY = 0;
    let animationFrameId = null;

    function updateGlowPosition() {
        const mainRect = homeMain.getBoundingClientRect();

        homeMain.style.setProperty("--glow-x", `${pointerX - mainRect.left}px`);
        homeMain.style.setProperty("--glow-y", `${pointerY - mainRect.top}px`);
        animationFrameId = null;
    }

    function scheduleGlowUpdate(event) {
        pointerX = event.clientX;
        pointerY = event.clientY;
        homeMain.classList.add("cursor-glow-active");

        if (animationFrameId === null) {
            animationFrameId = requestAnimationFrame(updateGlowPosition);
        }
    }

    homeMain.addEventListener("pointerenter", event => {
        scheduleGlowUpdate(event);
    });

    homeMain.addEventListener("pointermove", scheduleGlowUpdate);

    function hideGlow() {
        homeMain.classList.remove("cursor-glow-active");
    }

    homeMain.addEventListener("pointerleave", hideGlow);

    document.addEventListener("pointermove", event => {
        const mainRect = homeMain.getBoundingClientRect();
        const isOutsideMain =
            event.clientX < mainRect.left ||
            event.clientX > mainRect.right ||
            event.clientY < mainRect.top ||
            event.clientY > mainRect.bottom;

        if (isOutsideMain) {
            hideGlow();
        }
    }, { passive: true });

    window.addEventListener("blur", hideGlow);
}
