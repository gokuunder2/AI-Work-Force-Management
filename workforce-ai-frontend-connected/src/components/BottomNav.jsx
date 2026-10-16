import { Link, useLocation } from 'react-router-dom'

const items = [
  { icon: 'home', label: 'Home', to: '/' },
  { icon: 'group', label: 'Employees', to: '/employee-data' },
  { icon: 'task', label: 'Tasks', to: '/tasks' },
  { icon: 'settings', label: 'Settings', to: '#' },
]

export default function BottomNav() {
  const { pathname } = useLocation()

  return (
    <nav className="lg:hidden fixed bottom-0 w-full bg-surface-container border-t border-outline-variant flex justify-around items-center h-16 px-4 z-50">
      {items.map((item) => {
        const active = pathname === item.to
        return (
          <Link
            key={item.label}
            to={item.to}
            className={`flex flex-col items-center gap-1 p-2 active:scale-95 transition-transform ${
              active ? 'text-primary' : 'text-on-surface-variant'
            }`}
          >
            {active ? (
              <div className="bg-primary-container/20 rounded-full p-1">
                <span className="material-symbols-outlined">{item.icon}</span>
              </div>
            ) : (
              <span className="material-symbols-outlined">{item.icon}</span>
            )}
            <span className={`font-label-sm text-[10px] ${active ? 'font-bold' : ''}`}>{item.label}</span>
          </Link>
        )
      })}
    </nav>
  )
}
