import { expect, test, type Page } from '@playwright/test'

/** Plays the human seat with a simple rule: pass when allowed, otherwise lead the
 * lowest playable card as a single. Returns when the deal is over. */
async function playOut(page: Page) {
  const status = page.getByRole('status')
  for (let turn = 0; turn < 60; turn++) {
    const over = page.getByRole('dialog')
    await expect(status.filter({ hasText: /Your lead|Beat/ }).or(over)).toBeVisible({ timeout: 30_000 })
    if (await over.isVisible()) return
    const rows = page.getByRole('complementary').locator('.history .row')
    const before = await rows.count()
    const pass = page.getByRole('button', { name: 'Pass', exact: true })
    if (await pass.isEnabled()) {
      await pass.click()
    } else {
      const hand = page.getByRole('group', { name: 'Your hand' })
      await hand.locator('button.card:not(.dim)').first().click()
      await expect(hand.locator('button.card.selected')).toHaveCount(1)
      await page.getByRole('button', { name: 'Play', exact: true }).click()
    }
    // Our move is recorded before we continue.
    await expect.poll(() => rows.count(), { timeout: 10_000 }).toBeGreaterThan(before)
  }
  throw new Error('deal did not finish in 60 turns')
}

test('play a full deal against easy bots, with hints and the post-game review', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/?botDelay=0')

  // New-game dialog → easy bots
  await page.getByLabel('Opponents').selectOption('easy')
  await page.getByRole('button', { name: 'Deal' }).click()
  await expect(page.getByRole('group', { name: 'Your hand' }).locator('button.card')).toHaveCount(13, {
    timeout: 30_000,
  })

  // Study mode (Hints) shows up to three moves on our first turn; clicking one selects its cards.
  await expect(page.getByRole('status')).toHaveText(/Your lead|Beat/, { timeout: 30_000 })
  const moves = page.getByTestId('study-move')
  await expect(moves.first()).toBeVisible({ timeout: 30_000 })
  expect(await moves.count()).toBeLessThanOrEqual(3)
  await moves.first().click()
  const clear = page.getByRole('button', { name: 'Clear' })
  if (await clear.isEnabled()) await clear.click() // a "pass" hint selects no cards

  await playOut(page)

  // Results dialog: scores for all four players, then the review.
  const dialog = page.getByRole('dialog')
  await expect(dialog.getByRole('heading')).toHaveText(/went out first/)
  for (let p = 0; p < 4; p++) await expect(dialog.getByTestId(`result-${p}`)).toBeVisible()
  await dialog.getByRole('button', { name: 'Review my game' }).click()
  await expect(dialog.getByTestId('review').or(dialog.getByText('nothing to review'))).toBeVisible({
    timeout: 60_000,
  })
  await page.screenshot({ path: 'test-results/game-over.png' })

  // Escape closes the results; "Results" brings them back.
  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()
  await page.getByRole('button', { name: 'Results' }).click()
  await expect(dialog.getByRole('heading')).toHaveText(/went out first/)
  expect(errors).toEqual([])
})

test('quiz mode hides the solver until you move, then shows a verdict', async ({ page }) => {
  await page.goto('/?botDelay=0')
  await page.getByLabel('Opponents').selectOption('easy')
  await page.getByRole('button', { name: 'Deal' }).click()
  await page.getByRole('group', { name: 'Study mode setting' }).getByRole('button', { name: 'Quiz' }).click()
  await expect(page.getByRole('status')).toHaveText(/Your lead|Beat/, { timeout: 30_000 })
  await expect(page.getByTestId('study-move')).toHaveCount(0)
  await expect(page.getByText('Make your move')).toBeVisible()
  // Give the hidden analysis a moment, then move.
  await page.waitForTimeout(1500)
  const pass = page.getByRole('button', { name: 'Pass', exact: true })
  if (await pass.isEnabled()) await pass.click()
  else {
    await page.getByRole('group', { name: 'Your hand' }).locator('button.card:not(.dim)').first().click()
    await page.getByRole('button', { name: 'Play', exact: true }).click()
  }
  await expect(page.getByTestId('verdict')).toContainText('You played')
})

test('hints compare the solver with a network, and another network loads only when confirmed', async ({ page }) => {
  const config = await (await page.request.get('/api/config')).json()
  test.skip(config.models.length < 2, 'needs two trained networks in models/')
  const [first, second] = config.models as { id: string; label: string }[]

  await page.goto('/?botDelay=0')
  await page.getByLabel('Opponents').selectOption('easy')
  await page.getByLabel('Network').selectOption(first.id)
  await page.getByRole('button', { name: 'Deal' }).click()

  // Side by side: the solver's moves, the network's moves, and an agree/disagree verdict.
  const compare = page.getByTestId('compare')
  await expect(compare).toBeVisible({ timeout: 30_000 })
  await expect(compare.getByRole('heading', { name: first.label })).toBeVisible()
  expect(await compare.getByTestId('study-move').count()).toBeGreaterThan(0)
  expect(await compare.getByTestId('network-move').count()).toBeGreaterThan(0)
  await expect(page.getByTestId('network-pick')).toHaveText(/Agree|Disagree/)

  // Choosing another network only stages it; the hint changes after "Load".
  const picker = page.getByRole('combobox', { name: 'Network' })
  await picker.selectOption(second.id)
  await expect(compare.getByRole('heading', { name: first.label })).toBeVisible()
  await page.getByRole('button', { name: /^(Load|Use)$/ }).click()
  await expect(compare.getByRole('heading', { name: second.label })).toBeVisible({ timeout: 30_000 })
  await page.screenshot({ path: 'test-results/compare.png' })
})
