// reveal on scroll, active section in the nav, autoplay videos only while visible, copy BibTeX
(function () {
  var io = "IntersectionObserver" in window;
  var blocks = document.querySelectorAll(".statement, .card, .flow li, .chart, .split figure, .split2 > div, .table-wrap");
  if (io) {
    var rv = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); rv.unobserve(e.target); } });
    }, { threshold: 0.12 });
    blocks.forEach(function (b) { b.classList.add("reveal"); rv.observe(b); });

    var links = {};
    document.querySelectorAll(".nav nav a").forEach(function (a) { links[a.getAttribute("href").slice(1)] = a; });
    var sec = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        var a = links[e.target.id];
        if (a && e.isIntersecting) {
          Object.values(links).forEach(function (x) { x.classList.remove("on"); });
          a.classList.add("on");
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll("main section[id]").forEach(function (s) { sec.observe(s); });

    var vis = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        var v = e.target;
        if (!v.autoplay) return;
        if (e.isIntersecting) { var p = v.play(); if (p && p.catch) p.catch(function () {}); } else { v.pause(); }
      });
    });
    document.querySelectorAll("video[autoplay]").forEach(function (v) { vis.observe(v); });
  }
  document.querySelectorAll("[data-copy]").forEach(function (b) {
    b.addEventListener("click", function () {
      var t = document.querySelector(b.getAttribute("data-copy")).textContent;
      var done = function () { b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy BibTeX"; }, 1500); };
      if (navigator.clipboard) navigator.clipboard.writeText(t).then(done, function () {});
    });
  });
})();
