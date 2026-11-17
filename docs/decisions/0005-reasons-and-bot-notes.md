# 0005 Reasons for every priced move; the bot-notes entry

- The phase 3 bot-notes entry goes into hk-games through the orchestrator's checkout, not a big2 worker (hk-games 877b6f9). (Brian, 2026-11-17: "yes")
- The bot gives a reason for a move wherever the coach can today only state the points (hk-games D115; plan Task 35). (Brian, 2026-11-17: "yes", to "Coach: should the game bots give reasons? ... Recommend: yes, starting with Big2 and 13 cards")
- The win-chance bot (plan Task 34) starts now and may run ahead of poker's jobs. (Brian, 2026-11-17: "big2 highest-winrate bot can start now, since we've not decided what to do with mahjong and doudizhu. It can come before poker")
- Order of work across projects: Dou Dizhu's search solver first, this project's win-chance bot (Task 34) second, the Dou Dizhu test runs third. (Brian, 2026-11-17: "update priorities: new solver for doudizhu using search method, then big2 highest-winrate-bot, then 5 test runs for doudizhu. Put aside resuming the doudizhu and mahjong self-play runs for now")
- The order is confirmed and the job queue now follows it: this project's jobs (193 label, 194 train, 195 match; they replace 187, 188, 190) run ahead of the Dou Dizhu test jobs (196-198). (Brian, 2026-11-17: "swap so big2 gets second priority", "swap so test runs get third priority")
