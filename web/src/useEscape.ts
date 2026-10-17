import { useEffect } from 'react'

/** Calls `onEscape` when Escape is pressed, while the component is mounted (no-op if undefined). */
export function useEscape(onEscape: (() => void) | undefined) {
  useEffect(() => {
    if (!onEscape) return
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onEscape!()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onEscape])
}
