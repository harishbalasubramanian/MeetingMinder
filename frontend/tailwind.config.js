/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'blue': {
          50: '#f0f9ff',
          600: '#2563eb',
        },
        'purple': {
          600: '#9333ea',
        },
      },
    },
  },
  plugins: [],
  safelist: [
    { pattern: /bg-\[#(34D399|DC2626|2563EB|9CA3AF)20\]/ },
    { pattern: /text-\[#(34D399|DC2626|2563EB|9CA3AF)\]/ }
  ]
}


