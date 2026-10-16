import { Routes, Route } from 'react-router-dom'
import { Show, SignIn, useAuth } from '@clerk/react'
import { setTokenGetter } from './api.js'
import Home from './pages/Home.jsx'
import EmployeeData from './pages/EmployeeData.jsx'
import Tasks from './pages/Tasks.jsx'


export default function App() {
  const { getToken } = useAuth()
  setTokenGetter(getToken)
  return (
    <>
      {/* Signed out: only the sign-in screen */}
      <Show when="signed-out">
        <div
          style={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#0d0d0d',
          }}
        >
          <SignIn routing="hash" />
        </div>
      </Show>

      {/* Signed in: the full app */}
      <Show when="signed-in">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/employee-data" element={<EmployeeData />} />
          <Route path="/tasks" element={<Tasks />} />
        </Routes>
      </Show>
    </>
  )
}