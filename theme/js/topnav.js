(function () {
  "use strict";

  var root = typeof path_to_root !== "undefined" ? path_to_root : "";
  var STORAGE_KEY = "rb-theme";
  var media = window.matchMedia("(prefers-color-scheme: dark)");

  function resolve(pref) {
    return pref === "auto" ? (media.matches ? "dark" : "light") : pref;
  }

  function apply(pref) {
    var resolved = resolve(pref);
    document.documentElement.classList.remove("rb-light", "rb-dark");
    document.documentElement.classList.add("rb-" + resolved);

    var buttons = document.querySelectorAll(".rb-theme-switch button");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].classList.toggle("rb-active", buttons[i].dataset.theme === pref);
    }
  }

  function setPreference(pref) {
    localStorage.setItem(STORAGE_KEY, pref);
    apply(pref);
  }

  var nav = document.createElement("nav");
  nav.className = "rb-topnav";
  nav.innerHTML =
    '<div class="rb-topnav-inner">' +
      '<a class="rb-brand" href="' + root + 'index.html">rustebra</a>' +
      '<div class="rb-topnav-links">' +
        '<a href="' + root + 'index.html">Home</a>' +
        '<a href="' + root + 'algorithms/index.html">Algorithms</a>' +
        '<a href="' + root + 'adr/index.html">Architecture Decisions</a>' +
        '<a href="https://github.com/tec-eli/rustebra" target="_blank" rel="noopener noreferrer">GitHub</a>' +
        '<a class="rb-cta" href="' + root + 'rustdoc/rustebra/index.html">API Reference</a>' +
      "</div>" +
      '<div class="rb-theme-switch" role="group" aria-label="Theme">' +
        '<button type="button" data-theme="light">Light</button>' +
        '<button type="button" data-theme="dark">Dark</button>' +
        '<button type="button" data-theme="auto">Auto</button>' +
      "</div>" +
    "</div>";

  document.body.insertBefore(nav, document.body.firstChild);

  var sidebarAnchor = document.getElementById("mdbook-sidebar-toggle-anchor");
  if (sidebarAnchor) sidebarAnchor.checked = false;
  document.documentElement.classList.remove("sidebar-visible");
  try { localStorage.removeItem("mdbook-sidebar"); } catch (e) { }

  var buttons = nav.querySelectorAll(".rb-theme-switch button");
  for (var i = 0; i < buttons.length; i++) {
    buttons[i].addEventListener("click", function () {
      setPreference(this.dataset.theme);
    });
  }

  var stored = localStorage.getItem(STORAGE_KEY) || "auto";
  apply(stored);

  media.addEventListener("change", function () {
    if ((localStorage.getItem(STORAGE_KEY) || "auto") === "auto") {
      apply("auto");
    }
  });
})();
