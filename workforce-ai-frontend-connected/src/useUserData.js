import { useState, useEffect, useCallback } from 'react'
import { useUser } from '@clerk/react'

const EMPTY = { employees: [], tasks: [] }

export default function useUserData() {
    const { user, isLoaded } = useUser()
    const [data, setData] = useState(EMPTY)
    const key = user ? `workforce-data-${user.id}` : null

    // Load this user's data whenever the logged-in user changes
    useEffect(() => {
        if (!key) {
            setData(EMPTY)
            return
        }
        try {
            const saved = localStorage.getItem(key)
            setData(saved ? JSON.parse(saved) : EMPTY)
        } catch {
            setData(EMPTY)
        }
    }, [key])

    // Save new data for this user
    const save = useCallback(
        (next) => {
            setData(next)
            if (key) localStorage.setItem(key, JSON.stringify(next))
        },
        [key]
    )

    return { data, save, isLoaded, user }
}