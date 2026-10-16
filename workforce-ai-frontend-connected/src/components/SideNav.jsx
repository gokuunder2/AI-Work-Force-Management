import { Link } from 'react-router-dom'

const mainLinks = [
  { icon: 'group', label: 'Directory', key: 'directory', to: '/employee-data' },
  { icon: 'work', label: 'Recruitment', key: 'recruitment', to: '/tasks' },
]

export default function SideNav({ active }) {
  return (
    <aside className="flex flex-col h-full">
      <div className="px-6 mb-6">
        <button className="w-full bg-primary-container text-on-primary-container font-label-md text-label-md py-2 rounded hover:bg-primary transition-colors flex items-center justify-center gap-2">
          <span className="material-symbols-outlined text-[18px]">add</span>
          New Request
        </button>
      </div>

      <nav className="flex-1 px-2 space-y-1">
        {mainLinks.map((link) => {
          const isActive = link.key === active

          const className = isActive
            ? 'flex items-center gap-3 px-4 py-3 text-primary bg-secondary-container/10 border-l-4 border-primary hover:bg-surface-container-high transition-all rounded-r-full active:scale-95 font-label-md text-label-md'
            : 'flex items-center gap-3 px-4 py-3 text-on-surface-variant hover:bg-surface-container-high transition-all rounded-r-full active:scale-95 font-label-md text-label-md'

          return (
            <Link
              key={link.key}
              to={link.to}
              className={className}
            >
              <span
                className="material-symbols-outlined"
                style={
                  isActive
                    ? { fontVariationSettings: "'FILL' 1" }
                    : undefined
                }
              >
                {link.icon}
              </span>

              {link.label}
            </Link>
          )
        })}
      </nav>

      <div className="px-2 mt-auto space-y-1">
        <a
          href="#"
          onClick={(e) => e.preventDefault()}
          className="flex items-center gap-3 px-4 py-3 text-on-surface-variant hover:bg-surface-container-high transition-all rounded-r-full active:scale-95 font-label-md text-label-md"
        >
          <span className="material-symbols-outlined">contact_support</span>
          Support
        </a>

        <a
          href="#"
          onClick={(e) => e.preventDefault()}
          className="flex items-center gap-3 px-4 py-3 text-error hover:bg-surface-container-high transition-all rounded-r-full active:scale-95 font-label-md text-label-md"
        >
          <span className="material-symbols-outlined">logout</span>
          Logout
        </a>
      </div>
    </aside>
  )
}