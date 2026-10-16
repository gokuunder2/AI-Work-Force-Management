import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import TopNav from '../components/TopNav.jsx'
import BottomNav from '../components/BottomNav.jsx'
import { api, setTokenGetter } from '../api.js'
import { useUser, useAuth } from '@clerk/react'


export default function Home() {

  const { user } = useUser()
  const { getToken } = useAuth()
  setTokenGetter(getToken)


  const [employeeCount, setEmployeeCount] = useState(null)
  const [taskCount, setTaskCount] = useState(null)
  const [backendOnline, setBackendOnline] = useState(null)

  // Upload state
  const fileInputRef = useRef(null)
  const [uploadStatus, setUploadStatus] = useState('idle') // idle | uploading | success | error
  const [uploadMessage, setUploadMessage] = useState('')

  const taskFileInputRef = useRef(null)
  const [taskUploadStatus, setTaskUploadStatus] = useState('idle')
  const [taskUploadMessage, setTaskUploadMessage] = useState('')

  const loadStats = () =>
    Promise.all([api.listEmployees(), api.listTasks(), api.health()])
      .then(([employees, tasks]) => {
        setEmployeeCount(employees.count)
        setTaskCount(tasks.count)
        setBackendOnline(true)
      })
      .catch(() => setBackendOnline(false))

  useEffect(() => {
    loadStats()
  }, [])

  const handleFileChosen = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    try {
      setUploadStatus('uploading')
      setUploadMessage(`Uploading ${file.name}…`)
      const result = await api.uploadEmployees(file)
      setUploadStatus('success')
      setUploadMessage(result?.message || `${file.name} uploaded successfully.`)
      loadStats() // refresh the "Total Employees" number
    } catch (err) {
      setUploadStatus('error')
      setUploadMessage(err.message)
    } finally {
      e.target.value = '' // allow re-selecting the same file
    }
  }

  const handleTaskFileChosen = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    try {
      setTaskUploadStatus('uploading')
      setTaskUploadMessage(`Uploading ${file.name}…`)
      const result = await api.uploadTasks(file)
      setTaskUploadStatus('success')
      setTaskUploadMessage(result?.message || `${file.name} uploaded successfully.`)
      loadStats() // refresh the "Open Tasks" number
    } catch (err) {
      setTaskUploadStatus('error')
      setTaskUploadMessage(err.message)
    } finally {
      e.target.value = ''
    }
  }
  return (
    <div className="pt-16 pb-20 lg:pb-0 min-h-screen bg-surface-container-lowest">
      <TopNav />

      <main className="max-w-container-max mx-auto px-margin-mobile md:px-margin-desktop py-8 md:py-12 flex flex-col gap-gutter">
        {/* Hero */}
        <header className="flex flex-col gap-2">
          <h1 className="font-display-lg text-display-lg text-on-surface tracking-tight">
            Welcome, {user?.firstName || 'there'}.
          </h1>
          <p className="font-body-lg text-body-lg text-on-surface-variant max-w-2xl">
            Here is the current overview of your workforce. System operations are optimal.
          </p>
          {employeeCount === 0 && (
            <p className="font-body-md text-body-md text-on-surface-variant">
              No data uploaded yet. Upload an employee file to get started.
            </p>
          )}
        </header>

        {/* Bento grid */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-gutter mt-4">
          {/* Employee Data Upload */}
          <div className="md:col-span-8 bg-surface-container rounded-xl p-8 inner-glow border border-outline-variant hover:border-secondary transition-colors duration-300 group flex flex-col justify-between min-h-[320px] relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-secondary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />

            <div className="relative z-10 flex flex-col gap-4">
              <div className="w-12 h-12 rounded-lg bg-surface-container-lowest border border-outline-variant flex items-center justify-center text-secondary">
                <span className="material-symbols-outlined text-[24px]">cloud_upload</span>
              </div>
              <div>
                <h2 className="font-headline-lg text-headline-lg text-on-surface group-hover:text-secondary transition-colors">
                  Employee Data Upload
                </h2>
                <p className="font-body-md text-body-md text-on-surface-variant mt-2 max-w-md">
                  Batch upload new employee records, update existing profiles, and synchronize HR databases.
                </p>
              </div>
            </div>

            {/* Hidden file input, opened by the Upload button */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xlsx,.xls"
              className="hidden"
              onChange={handleFileChosen}
            />

            <div className="relative z-10 flex flex-col gap-3 mt-8">
              {uploadMessage && (
                <p
                  className={`font-body-md text-body-md ${uploadStatus === 'error'
                    ? 'text-red-500'
                    : uploadStatus === 'success'
                      ? 'text-secondary'
                      : 'text-on-surface-variant'
                    }`}
                >
                  {uploadMessage}
                </p>
              )}

              <div className="flex flex-wrap items-center gap-4">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploadStatus === 'uploading'}
                  className="flex items-center px-4 py-2 bg-secondary text-on-secondary rounded font-label-md text-label-md disabled:opacity-50 hover:opacity-90 transition-opacity"
                >
                  <span className="material-symbols-outlined mr-2 text-[18px]">upload_file</span>
                  {uploadStatus === 'uploading' ? 'Uploading…' : 'Upload File'}
                </button>

                <Link
                  to="/employee-data"
                  className="flex items-center text-secondary font-label-md text-label-md hover:translate-x-1 transition-transform duration-300"
                >
                  View Employees
                  <span className="material-symbols-outlined ml-2 text-[16px]">arrow_forward</span>
                </Link>
              </div>
            </div>
          </div>

          {/* Task Allocation */}
          <div className="md:col-span-4 bg-surface-container rounded-xl p-8 inner-glow border border-outline-variant hover:border-primary-container transition-colors duration-300 group flex flex-col justify-between min-h-[320px] relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary-container/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />

            <div className="relative z-10 flex flex-col gap-4">
              <div className="w-12 h-12 rounded-lg bg-surface-container-lowest border border-outline-variant flex items-center justify-center text-primary-container">
                <span className="material-symbols-outlined text-[24px]">assignment</span>
              </div>
              <div>
                <h2 className="font-headline-md text-headline-md text-on-surface group-hover:text-primary-container transition-colors">
                  Task Allocation
                </h2>
                <p className="font-body-md text-body-md text-on-surface-variant mt-2">
                  Upload task files, distribute and monitor departmental assignments.
                </p>
              </div>
            </div>

            {/* Hidden file input, opened by the Upload button */}
            <input
              ref={taskFileInputRef}
              type="file"
              accept=".csv,.xlsx,.xls"
              className="hidden"
              onChange={handleTaskFileChosen}
            />

            <div className="relative z-10 flex flex-col gap-3 mt-8">
              {taskUploadMessage && (
                <p
                  className={`font-body-md text-body-md ${taskUploadStatus === 'error'
                    ? 'text-red-500'
                    : taskUploadStatus === 'success'
                      ? 'text-primary-container'
                      : 'text-on-surface-variant'
                    }`}
                >
                  {taskUploadMessage}
                </p>
              )}

              <div className="flex flex-wrap items-center gap-4">
                <button
                  type="button"
                  onClick={() => taskFileInputRef.current?.click()}
                  disabled={taskUploadStatus === 'uploading'}
                  className="flex items-center px-4 py-2 bg-primary-container text-on-primary-container rounded font-label-md text-label-md disabled:opacity-50 hover:opacity-90 transition-opacity"
                >
                  <span className="material-symbols-outlined mr-2 text-[18px]">upload_file</span>
                  {taskUploadStatus === 'uploading' ? 'Uploading…' : 'Upload File'}
                </button>

                <Link
                  to="/tasks"
                  className="flex items-center text-primary-container font-label-md text-label-md hover:translate-x-1 transition-transform duration-300"
                >
                  Manage Tasks
                  <span className="material-symbols-outlined ml-2 text-[16px]">arrow_forward</span>
                </Link>
              </div>
            </div>
          </div>

          {/* Quick stats */}
          <div className="md:col-span-12 grid grid-cols-1 md:grid-cols-4 gap-gutter mt-4">

            {/*Total employees */}
            <div className="bg-surface-container-low rounded-xl p-6 inner-glow border border-outline-variant flex flex-col gap-2">
              <span className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wider">
                Total Employees
              </span>
              <div className="flex items-baseline gap-2">
                <span className="font-display-lg text-display-lg text-on-surface">
                  {employeeCount === null ? '—' : employeeCount.toLocaleString()}
                </span>
                <span className="text-secondary bg-secondary/10 px-2 py-0.5 rounded text-label-sm font-label-sm flex items-center">
                  <span className="material-symbols-outlined text-[14px] mr-1">groups</span> On file
                </span>
              </div>
            </div>

            {/*Open tasks */}
            <div className="bg-surface-container-low rounded-xl p-6 inner-glow border border-outline-variant flex flex-col gap-2">
              <span className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wider">
                Open Tasks
              </span>
              <div className="flex items-baseline gap-2">
                <span className="font-display-lg text-display-lg text-on-surface">
                  {taskCount === null ? '—' : taskCount.toLocaleString()}
                </span>
                <span className="text-tertiary bg-tertiary/10 px-2 py-0.5 rounded text-label-sm font-label-sm">
                  Requires Action
                </span>
              </div>
            </div>

            {/* System status */}
            <div className="bg-surface-container-low rounded-xl p-6 inner-glow border border-outline-variant flex flex-col gap-2 md:col-span-2 relative overflow-hidden">
              <div className="relative z-10 flex justify-between items-center h-full">
                <div className="flex flex-col gap-2">
                  <span className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wider">
                    System Status
                  </span>
                  <span className="font-headline-md text-headline-md text-on-surface flex items-center gap-2">
                    <span
                      className={`w-3 h-3 rounded-full inline-block ${backendOnline ? 'bg-secondary animate-pulse' : backendOnline === false ? 'bg-red-500' : 'bg-outline'
                        }`}
                    />
                    {backendOnline === null && 'Checking backend…'}
                    {backendOnline === true && 'All Systems Operational'}
                    {backendOnline === false && 'Backend Unreachable'}
                  </span>
                </div>
                <button className="px-4 py-2 bg-transparent border border-outline-variant text-on-surface rounded font-label-md text-label-md hover:bg-surface-container-highest transition-colors">
                  View Logs
                </button>
              </div>
              <div className="absolute -right-8 -bottom-8 w-32 h-32 bg-primary/5 rounded-full blur-2xl" />
            </div>

          </div>
        </div>
      </main>

      <BottomNav />
    </div>
  )
}
