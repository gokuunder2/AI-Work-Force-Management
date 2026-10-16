import { Link, useLocation } from 'react-router-dom'
import { UserButton } from '@clerk/react'

const links = [
  { to: '/', label: 'Home' },
  { to: '/employee-data', label: 'Employee Data' },
  { to: '/tasks', label: 'Tasks' },
]

export default function TopNav({ searchPlaceholder = 'Search...' }) {
  const { pathname } = useLocation()

  return (
    <header className="fixed top-0 w-full z-50 bg-surface-container border-b border-outline-variant shadow-sm">
      <div className="flex justify-between items-center px-margin-mobile md:px-margin-desktop h-16 max-w-container-max mx-auto">
        <div className="flex items-center gap-6">
          <Link
            to="/"
            className="font-headline-md text-headline-md font-bold text-primary cursor-pointer active:opacity-80"
          >
            Workforce AI
          </Link>
        </div>

        <nav className="hidden md:flex gap-8 items-center">
          {links.map((link) => {
            const active = pathname === link.to
            return (
              <Link
                key={link.to}
                to={link.to}
                className={
                  active
                    ? 'text-primary border-b-2 border-primary pb-1 font-label-md text-label-md transition-colors duration-200 cursor-pointer active:opacity-80'
                    : 'text-on-surface-variant font-label-md text-label-md hover:text-primary transition-colors duration-200 cursor-pointer active:opacity-80'
                }
              >
                {link.label}
              </Link>
            )
          })}
        </nav>

        <div className="flex items-center gap-4 text-on-surface-variant">
          <button
            aria-label="notifications"
            className="hover:text-primary transition-colors cursor-pointer active:opacity-80"
          >
            <span className="material-symbols-outlined">notifications</span>
          </button>
          <button aria-label="help" className="hover:text-primary transition-colors cursor-pointer active:opacity-80">
            <span className="material-symbols-outlined">help</span>
          </button>
          <button
            aria-label="settings"
            className="hover:text-primary transition-colors cursor-pointer active:opacity-80"
          >
            <span className="material-symbols-outlined">settings</span>
          </button>
          <div className="ml-2 flex items-center">
            <UserButton />
          </div>
        </div>
      </div>
    </header>
  )
}