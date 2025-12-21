(function () {
  var root = document.documentElement;

  function setThemeClass(enableLight) {
    if (enableLight) {
      root.classList.add('theme-light');
    } else {
      root.classList.remove('theme-light');
    }
    try {
      localStorage.setItem('preferredTheme', enableLight ? 'light' : 'dark');
    } catch (e) {
      // ignore storage issues
    }
    updateButtons(enableLight);
  }

  function updateButtons(isLight) {
    var buttons = document.querySelectorAll('.theme-toggle');
    buttons.forEach(function (btn) {
      btn.textContent = isLight ? '深色模式' : '浅色模式';
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    var toggleButtons = document.querySelectorAll('.theme-toggle');
    if (!toggleButtons.length) {
      return;
    }
    var initialLight = root.classList.contains('theme-light');
    updateButtons(initialLight);

    toggleButtons.forEach(function (button) {
      button.addEventListener('click', function () {
        var nextIsLight = !root.classList.contains('theme-light');
        setThemeClass(nextIsLight);
      });
    });
  });
})();
