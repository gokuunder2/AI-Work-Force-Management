import { useEffect, useState } from 'react'
import TopNav from '../components/TopNav.jsx'
import SideNav from '../components/SideNav.jsx'
import BottomNav from '../components/BottomNav.jsx'
import { api, DEPARTMENTS } from '../api.js'

const priorityBadge = {
  High: 'bg-tertiary-container/10 text-tertiary-container border border-tertiary-container/20',
  Medium: 'bg-secondary/10 text-secondary border border-secondary/20',
  Low: 'bg-surface-variant text-on-surface-variant border border-outline-variant',
}

const accentByComplexity = {
  High: 'bg-tertiary-container',
  Medium: 'bg-secondary',
  Low: 'bg-outline-variant',
}

function viewTransform(view, name) {
  if (view === name) return 'translate-x-0'
  if (name === 'list') return '-translate-x-full'
  return 'translate-x-full'
}

function initialsFor(n) {
  return String(n).slice(-2)
}

export default function Tasks() {
  const [view, setView] = useState('list') // 'list' | 'recommend' | 'bulk'

  // Task list state
  const [tasks, setTasks] = useState([])
  const [tasksLoading, setTasksLoading] = useState(true)
  const [tasksError, setTasksError] = useState(null)
  const [deptFilter, setDeptFilter] = useState('')

  // Recommend view state
  const [recDept, setRecDept] = useState(DEPARTMENTS[0])
  const [recDesc, setRecDesc] = useState('')
  const [recResults, setRecResults] = useState(null)
  const [recLoading, setRecLoading] = useState(false)
  const [recError, setRecError] = useState(null)

  // Bulk allocation view state
  const [allocResult, setAllocResult] = useState(null)
  const [allocLoading, setAllocLoading] = useState(false)
  const [allocError, setAllocError] = useState(null)

  function loadTasks(department) {
    setTasksLoading(true)
    api
      .listTasks(department || undefined)
      .then((res) => {
        setTasks(res.data)
        setTasksError(null)
      })
      .catch((err) => setTasksError(err.message))
      .finally(() => setTasksLoading(false))
  }

  useEffect(() => {
    loadTasks()
  }, [])

  function runRecommend(e) {
    e.preventDefault()
    if (!recDesc.trim()) return
    setRecLoading(true)
    setRecError(null)
    api
      .recommend(recDept, recDesc.trim())
      .then((res) => {
        if (res.status === 'Error') {
          setRecError(res.message)
          setRecResults(null)
        } else {
          setRecResults(res.data)
        }
      })
      .catch((err) => setRecError(err.message))
      .finally(() => setRecLoading(false))
  }

  function runFullAllocation() {
    setAllocLoading(true)
    setAllocError(null)
    api
      .runAllocation()
      .then((res) => setAllocResult(res.data))
      .catch((err) => setAllocError(err.message))
      .finally(() => setAllocLoading(false))
  }

  return (
    <div className="min-h-screen flex flex-col bg-surface-container-lowest text-on-background">
      <TopNav searchPlaceholder="Search tasks..." />
      <SideNav active="recruitment" />

      <main className="flex-1 w-full pt-20 md:pt-24 pb-24 lg:pb-8 px-margin-mobile md:px-margin-desktop lg:ml-64 max-w-container-max mx-auto">
        {/* Header & breadcrumbs */}
        <div className="mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-on-surface-variant font-label-sm text-label-sm mb-2">
              <button
                aria-label="Back to list"
                onClick={() => setView('list')}
                className="material-symbols-outlined text-[16px] cursor-pointer hover:text-primary transition-colors"
              >
                arrow_back
              </button>
              <span className="cursor-pointer hover:text-primary transition-colors">Dashboard</span>
              <span className="material-symbols-outlined text-[16px]">chevron_right</span>
              <span className="text-primary">Task Management</span>
            </div>
            <h2 className="font-headline-lg text-headline-lg font-bold text-on-surface">Task Allocation Flow</h2>
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => setView('recommend')}
              className="bg-surface border border-outline-variant hover:border-primary text-primary font-label-md text-label-md px-4 py-2 rounded flex items-center gap-2 transition-all shadow-[0_0_10px_rgba(77,142,255,0.1)] focus:outline-none focus:ring-1 focus:ring-primary"
            >
              <span className="material-symbols-outlined text-[18px]">auto_awesome</span> Top Recommend
            </button>
            <button
              onClick={() => {
                setView('bulk')
                if (!allocResult) runFullAllocation()
              }}
              className="bg-secondary/10 hover:bg-secondary/20 border border-secondary/30 text-secondary font-label-md text-label-md px-4 py-2 rounded flex items-center gap-2 transition-all focus:outline-none focus:ring-1 focus:ring-secondary"
            >
              <span className="material-symbols-outlined text-[18px]">grid_view</span> Bulk Allocation
            </button>
          </div>
        </div>

        {/* Dynamic views */}
        <div className="relative w-full min-h-[600px] overflow-hidden" id="views-container">
          {/* View 1: Task list */}
          <div
            className={`absolute inset-0 transition-transform duration-500 ease-in-out transform ${viewTransform(
              view,
              'list',
            )} w-full flex flex-col gap-6`}
          >
            <div className="bg-surface-container-high rounded-xl p-6 inner-glow border border-outline-variant/30 flex flex-col md:flex-row gap-4 items-center">
              <div className="flex-1 w-full relative">
                <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-on-surface-variant">
                  filter_list
                </span>
                <span className="pl-12 font-body-md text-body-md text-on-surface-variant">
                  {tasksLoading ? 'Loading tasks…' : `${tasks.length} task${tasks.length === 1 ? '' : 's'} on file`}
                  {tasksError && ` — couldn't reach backend (${tasksError})`}
                </span>
              </div>
              <div className="flex gap-3 w-full md:w-auto">
                <select
                  value={deptFilter}
                  onChange={(e) => {
                    setDeptFilter(e.target.value)
                    loadTasks(e.target.value || undefined)
                  }}
                  className="bg-surface-container-lowest border border-outline-variant/50 rounded-lg py-3 px-4 text-on-surface font-label-md text-label-md focus:border-primary focus:ring-1 focus:ring-primary outline-none"
                >
                  <option value="">All Departments</option>
                  {DEPARTMENTS.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 auto-rows-max overflow-y-auto pr-2 pb-4"
              style={{ maxHeight: 'calc(100% - 100px)' }}
            >
              {tasksLoading && <p className="font-body-sm text-body-sm text-on-surface-variant">Loading…</p>}
              {!tasksLoading &&
                tasks.map((task) => (
                  <div
                    key={task.task_id}
                    className="bg-surface-container-high rounded-xl p-6 inner-glow border border-outline-variant/30 flex flex-col gap-4 hover:border-outline-variant transition-colors group cursor-pointer relative overflow-hidden"
                  >
                    <div className={`absolute top-0 left-0 w-1 h-full ${accentByComplexity[task.task_complexity]}`} />
                    <div className="flex justify-between items-start">
                      <span className={`px-2 py-1 text-label-sm font-label-sm rounded ${priorityBadge[task.task_priority]}`}>
                        {task.task_priority} Priority
                      </span>
                      {task.criticality === 'Critical' && (
                        <span className="material-symbols-outlined text-red-400 text-[18px]" title="Critical">
                          priority_high
                        </span>
                      )}
                    </div>
                    <div>
                      <h3 className="font-headline-md text-headline-md text-on-surface mb-1">{task.task_name}</h3>
                      <p className="font-body-sm text-body-sm text-on-surface-variant line-clamp-2">
                        {task.department} • {task.target_role} — needs: {task.required_skills}
                      </p>
                    </div>
                    <div className="mt-auto pt-4 border-t border-outline-variant/20 flex items-center justify-between">
                      <span className="font-label-sm text-label-sm text-outline">{task.task_id}</span>
                      <span className="font-label-sm text-label-sm text-outline flex items-center gap-1">
                        <span className="material-symbols-outlined text-[14px]">schedule</span> {task.estimated_hours}h • {task.task_duration}
                      </span>
                    </div>
                  </div>
                ))}
            </div>
          </div>

          {/* View 2: Top Recommend */}
          <div
            className={`absolute inset-0 transition-transform duration-500 ease-in-out transform ${viewTransform(
              view,
              'recommend',
            )} w-full h-full flex flex-col gap-6 bg-surface-container-lowest z-10`}
          >
            <div className="bg-surface-container-high rounded-xl p-6 inner-glow border border-primary/20 flex-1 flex flex-col shadow-[0_4px_30px_rgba(77,142,255,0.05)]">
              <div className="flex justify-between items-center mb-6 flex-wrap gap-4">
                <h3 className="font-headline-lg text-headline-lg text-on-surface flex items-center gap-3">
                  <span className="material-symbols-outlined text-primary">temp_preferences_custom</span>
                  AI Match Results
                </h3>
              </div>

              <form onSubmit={runRecommend} className="flex flex-col md:flex-row gap-3 mb-6">
                <select
                  value={recDept}
                  onChange={(e) => setRecDept(e.target.value)}
                  className="bg-surface-container-lowest border border-outline-variant/50 rounded-lg py-3 px-4 text-on-surface font-label-md text-label-md focus:border-primary focus:ring-1 focus:ring-primary outline-none"
                >
                  {DEPARTMENTS.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
                <input
                  value={recDesc}
                  onChange={(e) => setRecDesc(e.target.value)}
                  placeholder="Describe the task, e.g. 'Needs strong Python and data visualization skills'"
                  className="flex-1 bg-surface-container-lowest border border-outline-variant/50 rounded-lg py-3 px-4 text-on-surface placeholder:text-outline focus:border-primary focus:ring-1 focus:ring-primary transition-all font-body-md text-body-md outline-none"
                />
                <button
                  type="submit"
                  disabled={recLoading}
                  className="bg-primary hover:bg-primary/90 text-on-primary font-label-md text-label-md px-6 py-3 rounded-lg flex items-center gap-2 transition-colors whitespace-nowrap disabled:opacity-50"
                >
                  {recLoading ? 'Matching…' : 'Find Matches'}
                </button>
              </form>

              {recError && <p className="font-body-sm text-body-sm text-red-400 mb-4">{recError}</p>}

              <div className="flex-1 overflow-y-auto space-y-4 pr-2">
                {!recResults && !recError && (
                  <p className="font-body-sm text-body-sm text-on-surface-variant">
                    Pick a department and describe the task to see the best-matched employees.
                  </p>
                )}
                {recResults &&
                  recResults.map((rec, i) => (
                    <div
                      key={rec.EmployeeNumber}
                      className={`bg-surface-container-lowest border border-outline-variant/30 rounded-lg p-4 flex flex-col md:flex-row items-center gap-6 hover:border-primary/50 transition-colors ${i === 0 ? '' : 'opacity-80'
                        }`}
                    >
                      <div className="relative">
                        <div
                          className={`w-16 h-16 rounded-full flex items-center justify-center font-headline-md text-headline-md ${i === 0 ? 'bg-secondary/20 text-secondary' : 'bg-surface-variant text-on-surface-variant'
                            }`}
                        >
                          {initialsFor(rec.EmployeeNumber)}
                        </div>
                        {i === 0 && (
                          <div className="absolute -bottom-1 -right-1 bg-surface-container-lowest rounded-full p-1">
                            <div className="w-4 h-4 bg-secondary rounded-full" />
                          </div>
                        )}
                      </div>
                      <div className="flex-1 text-center md:text-left">
                        <h4 className="font-headline-md text-headline-md text-on-surface">
                          {rec.JobRole} #{rec.EmployeeNumber}
                        </h4>
                        <p className="font-body-sm text-body-sm text-on-surface-variant">
                          {rec.Department} • Performance {rec.PerformanceRating}/4
                        </p>
                        <div className="flex gap-2 mt-2 justify-center md:justify-start flex-wrap">
                          {rec.skills.split(',').slice(0, 3).map((s) => (
                            <span
                              key={s}
                              className="px-2 py-0.5 bg-surface-container-highest text-on-surface font-label-sm text-label-sm rounded"
                            >
                              {s.trim()}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div className="flex flex-col items-center gap-2">
                        <div className={`flex items-center gap-2 ${i === 0 ? 'text-secondary' : 'text-primary-fixed-dim'}`}>
                          <span className="material-symbols-outlined">check_circle</span>
                          <span className="font-headline-md text-headline-md">{Math.round(rec.Total_Score * 100)}% Match</span>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>

          {/* View 3: Bulk Allocation */}
          <div
            className={`absolute inset-0 transition-transform duration-500 ease-in-out transform ${viewTransform(
              view,
              'bulk',
            )} w-full h-full bg-surface-container-high rounded-xl inner-glow border border-outline-variant/30 p-6 flex flex-col z-10 overflow-hidden`}
          >
            <div className="flex justify-between items-center mb-6">
              <h3 className="font-headline-lg text-headline-lg text-on-surface flex items-center gap-3">
                <span className="material-symbols-outlined text-secondary">dataset</span>
                Bulk Allocation Results
              </h3>
              <button
                onClick={runFullAllocation}
                disabled={allocLoading}
                className="bg-secondary hover:bg-secondary/90 text-on-secondary font-label-md text-label-md px-6 py-2 rounded transition-colors shadow-lg disabled:opacity-50"
              >
                {allocLoading ? 'Allocating…' : 'Run Allocation'}
              </button>
            </div>

            {allocError && <p className="font-body-sm text-body-sm text-red-400 mb-4">{allocError}</p>}

            <div className="flex-1 overflow-auto border border-outline-variant/20 rounded bg-surface-container-lowest">
              {allocLoading && <p className="p-6 font-body-sm text-body-sm text-on-surface-variant">Running allocation across all tasks…</p>}
              {!allocLoading && allocResult && (
                <>
                  <div className="p-4 font-body-sm text-body-sm text-on-surface-variant border-b border-outline-variant/20">
                    {allocResult.assignments.length} assigned • {allocResult.unassigned.length} unassigned
                  </div>
                  <table className="w-full text-left border-collapse">
                    <thead className="bg-surface-container-high sticky top-0 z-20">
                      <tr>
                        <th className="p-4 font-label-md text-label-md text-on-surface border-b border-outline-variant/30">Task</th>
                        <th className="p-4 font-label-md text-label-md text-on-surface border-b border-outline-variant/30">Complexity</th>
                        <th className="p-4 font-label-md text-label-md text-on-surface border-b border-outline-variant/30">Assigned Employee</th>
                        <th className="p-4 font-label-md text-label-md text-on-surface border-b border-outline-variant/30 text-right">Match Score</th>
                      </tr>
                    </thead>
                    <tbody className="font-body-sm text-body-sm text-on-surface-variant">
                      {allocResult.assignments.map((a) => (
                        <tr key={a.Task_id} className="border-b border-outline-variant/10 hover:bg-surface-container-highest/50 transition-colors">
                          <td className="p-4">
                            <div className="text-on-surface">{a.Task_name}</div>
                            <div className="font-label-sm text-outline">{a.Task_id}</div>
                          </td>
                          <td className="p-4">{a.Complexity}</td>
                          <td className="p-4 flex items-center gap-3">
                            <div className="w-8 h-8 rounded-full bg-secondary/20 text-secondary flex items-center justify-center font-label-sm">
                              {initialsFor(a.EmployeeNumber)}
                            </div>
                            EMP-{a.EmployeeNumber}
                          </td>
                          <td className="p-4 text-right text-secondary font-label-md">{Math.round(a.Task_Allocation_score * 100)}%</td>
                        </tr>
                      ))}
                      {allocResult.unassigned.map((u) => (
                        <tr key={u.Task_id} className="border-b border-outline-variant/10 opacity-60">
                          <td className="p-4">
                            <div className="text-on-surface">{u.Task_name}</div>
                            <div className="font-label-sm text-outline">{u.Task_id}</div>
                          </td>
                          <td className="p-4" colSpan={2}>
                            {u.Reason}
                          </td>
                          <td className="p-4 text-right">—</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </div>
          </div>
        </div>
      </main>

      <BottomNav />
    </div>
  )
}
