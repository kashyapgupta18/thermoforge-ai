/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        graphite: {
          950: '#0B0F17',
          900: '#111827',
          850: '#161F30',
          800: '#1F293D',
          700: '#334155',
          600: '#475569'
        },
        forge: {
          cyan: '#06B6D4',
          blue: '#3B82F6',
          amber: '#F59E0B',
          emerald: '#10B981',
          rose: '#F43F5E'
        }
      }
    },
  },
  plugins: [],
}
