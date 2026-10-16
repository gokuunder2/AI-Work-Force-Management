// Base URL of the FastAPI backend. Override with VITE_API_URL in a .env file
// (project root, next to package.json) and restart `npm run dev` after editing it.
// Default uses 127.0.0.1 because `localhost` can resolve to IPv6 on some systems.
export const API_BASE = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '')

let getTokenFn = null
export const setTokenGetter = (fn) => {
  getTokenFn = fn
}

async function authHeaders() {
  try {
    const token = getTokenFn ? await getTokenFn() : null
    return token ? { Authorization: `Bearer ${token}` } : {}
  } catch {
    return {}
  }
}

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`
  const isFormData = options.body instanceof FormData

  const auth = await authHeaders()
  let res
  try {

    res = await fetch(url, {
      ...options,
      headers: {
        ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
        ...auth,
        ...options.headers,
      },
    })
  } catch (err) {
    // Network failure / CORS block / backend not running
    throw new Error(`Cannot reach backend at ${url}. Is the API running? (${err.message})`)
  }

  // If the server returned HTML (e.g. the Vite dev server's index.html), the URL is wrong.
  const contentType = res.headers.get('content-type') || ''
  if (!contentType.includes('application/json')) {
    throw new Error(
      `Expected JSON from ${url} but got ${contentType || 'unknown content type'} (status ${res.status})`
    )
  }

  const body = await res.json()

  if (!res.ok) {
    // FastAPI returns `detail` as a string, or as an array for validation (422) errors.
    let detail = res.statusText
    if (typeof body.detail === 'string') {
      detail = body.detail
    } else if (Array.isArray(body.detail)) {
      detail = body.detail
        .map((d) => `${(d.loc || []).join('.')}: ${d.msg}`)
        .join('; ')
    } else if (body.message) {
      detail = body.message
    }
    throw new Error(detail)
  }

  return body
}

// Builds a FormData body with the file under the given field name.
// The field name MUST match the parameter name in the FastAPI endpoint
// (check the request body in http://127.0.0.1:8000/docs).
function fileBody(file, fieldName = 'file') {
  const formData = new FormData()
  formData.append(fieldName, file)
  return formData
}

export const api = {
  health: () => request('/'),

  // ---- Employees ----
  // Optional paging: listEmployees(undefined, { limit: 500, offset: 0 })
  listEmployees: (department, { limit, offset } = {}) => {
    const params = new URLSearchParams()
    if (department) params.set('department', department)
    if (limit !== undefined) params.set('limit', limit)
    if (offset !== undefined) params.set('offset', offset)
    const qs = params.toString()
    return request(`/employees${qs ? `?${qs}` : ''}`)
  },

  // Fetches EVERY employee by requesting pages until the total is reached.
  // Returns { count, employees }.
  listAllEmployees: async (department) => {
    const pageSize = 500
    const extract = (res) =>
      Array.isArray(res) ? res : res?.employees || res?.data || res?.items || res?.results || []

    let all = []
    let total = Infinity
    let offset = 0

    while (all.length < total) {
      const res = await api.listEmployees(department, { limit: pageSize, offset })
      const rows = extract(res)
      if (rows.length === 0) break

      // Safety: if the backend ignores `offset`, it returns the same first page again.
      if (offset > 0 && JSON.stringify(rows[0]) === JSON.stringify(all[0])) break

      all = all.concat(rows)
      total = typeof res?.count === 'number' ? res.count : all.length
      if (rows.length < pageSize) break // last page
      offset += pageSize
    }

    return { count: Math.max(total === Infinity ? all.length : total, all.length), employees: all }
  },

  getEmployee: (employeeNumber) => request(`/employees/${employeeNumber}`),

  // Note the capital "E": FastAPI paths are case-sensitive.
  uploadEmployees: (file) =>
    request('/Employee/upload', { method: 'POST', body: fileBody(file) }),

  // ---- Tasks ----
  listTasks: (department) =>
    request(`/tasks${department ? `?department=${encodeURIComponent(department)}` : ''}`),

  getTask: (taskId) => request(`/tasks/${taskId}`),

  uploadTasks: (file) =>
    request('/task/upload', { method: 'POST', body: fileBody(file) }),

  // ---- ML / allocation ----
  predictAttrition: (employeeNumber) =>
    request(`/predict/${employeeNumber}`, { method: 'POST' }),

  runAllocation: (taskIds) =>
    request('/task-allocation', {
      method: 'POST',
      body: JSON.stringify(taskIds ? { task_ids: taskIds } : {}),
    }),

  recommend: (department, description) =>
    request('/recommend', {
      method: 'POST',
      body: JSON.stringify({ department, description }),
    }),
}

// The three departments that actually exist in the backend's dataset.
export const DEPARTMENTS = ['Human Resources', 'Research & Development', 'Sales']
