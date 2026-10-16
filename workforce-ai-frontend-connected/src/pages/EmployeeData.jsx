import { useEffect, useState } from 'react'
import TopNav from '../components/TopNav.jsx'
import BottomNav from '../components/BottomNav.jsx'
import { api } from '../api.js'

export default function EmployeeData() {
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = async () => {
    try {
      setLoading(true)
      setError('')
      const res = await api.listAllEmployees()
      setRows(res.employees)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // Runs every time the page opens, so previously uploaded data always shows
  useEffect(() => {
    load()
  }, [])

  // Build columns from the data itself (first 8 keys) so it works with any schema
  const columns = rows.length ? Object.keys(rows[0]).slice(0, 8) : []

  return (
    <div className="pt-16 pb-20 lg:pb-0 min-h-screen bg-surface-container-lowest">
      <TopNav />

      <main className="max-w-container-max mx-auto px-margin-mobile md:px-margin-desktop py-8 md:py-12 flex flex-col gap-6">
        <header className="flex items-center justify-between">
          <div>
            <h1 className="font-display-lg text-display-lg text-on-surface tracking-tight">Employee Data</h1>
            {!loading && !error && (
              <p className="text-on-surface-variant mt-1">{rows.length.toLocaleString()} records loaded</p>
            )}
          </div>
          <button
            onClick={load}
            disabled={loading}
            className="px-4 py-2 border border-outline-variant text-on-surface rounded font-label-md text-label-md hover:bg-surface-container-highest disabled:opacity-50"
          >
            {loading ? 'Loading…' : 'Refresh'}
          </button>
        </header>

        {error && (
          <div className="rounded-xl border border-red-900 bg-red-950/30 text-red-400 p-4">{error}</div>
        )}

        {!error && !loading && rows.length === 0 && (
          <p className="text-on-surface-variant">No employees yet. Upload a file from the Home page.</p>
        )}

        {rows.length > 0 && (
          <div className="overflow-auto max-h-[70vh] rounded-xl border border-outline-variant">
            <table className="w-full text-left text-on-surface">
              <thead className="bg-surface-container text-on-surface-variant uppercase text-label-sm sticky top-0 z-10">
                <tr>
                  {columns.map((c) => (
                    <th key={c} className="px-4 py-3 whitespace-nowrap bg-surface-container">
                      {c.replace(/_/g, ' ')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr
                    key={`${row.EmployeeNumber ?? row.employee_number ?? row.id ?? 'row'}-${i}`}
                    className="border-t border-outline-variant"
                  >
                    {columns.map((c) => (
                      <td key={c} className="px-4 py-3 whitespace-nowrap">
                        {String(row[c] ?? '')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>

      <BottomNav />
    </div>
  )
}
