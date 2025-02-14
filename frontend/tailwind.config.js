const defaultTheme = require('tailwindcss/defaultTheme')

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    screens: {
      'mini':   {'max': '384px'},
      'mobile': {'max': '999px'},
      'laptop': {'min': '1000px'},
      ...defaultTheme.screens
    },
    extend: {},
  },
  plugins: [],
}