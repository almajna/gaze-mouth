"""Synthetic (phoneme lattice, screen context) -> text pairs for training the reconstructor.

Realistic OS commands, terminal commands, developer prompts, chat replies and browser commands, plus
general-vocabulary drills (half of the data), are converted to ARPAbet and corrupted the way silent lip reading corrupts them: confusions inside a
viseme class (p/b/m, t/d/n, k/g/ng, f/v, s/z, ...), deleted unstressed vowels and H, near-vowel
swaps, and the odd inserted sound. The lattices use the schema of silent_avsr.PhonemeLattice.

This is text-derived data for the language model only. It says nothing about how well the visual
encoder reads lips.

  .venv-gf/bin/python -m context_decoder.synth_phoneme_data --out models/open_vocab_lip/synth --n 12000
"""
from hashlib import sha1
from pathlib import Path
import argparse
import json
import math
import random

from silent_avsr.lexicon import g2p
from silent_avsr.phonemes import PHONEMES, VISEME_CLASSES, VISEME_OF, VOWELS, homophenes, strip_stress

from .prompt import normalize_text
from .phoneme_search import common_words

FRAME_RATE = 30.0
TOP_K = 5

# Confusions that cross a viseme class: tongue position is hidden, lip rounding looks alike.
NEAR = {"L": ("N", "D", "T"), "R": ("W",), "W": ("R",), "Y": ("IY",), "NG": ("N",), "N": ("L",),
        "IH": ("EH",), "EH": ("IH",), "AH": ("ER", "IH"), "ER": ("AH",), "OW": ("AO", "UH"),
        "UW": ("OW",), "UH": ("OW",), "AY": ("AA",), "AW": ("AA", "OW"), "OY": ("AO",), "AE": ("AA",)}

# ---------------------------------------------------------------------------------------------
# Phrase sources
# ---------------------------------------------------------------------------------------------
APPS = ("safari", "chrome", "terminal", "finder", "mail", "messages", "slack", "notes", "calendar",
        "music", "spotify", "xcode", "preview", "photos", "system settings", "maps", "zoom", "discord",
        "figma", "notion", "calculator", "reminders", "facetime", "keynote", "pages", "numbers",
        "cursor", "activity monitor", "visual studio code", "the app store")
APP_VERBS = ("open", "close", "quit", "switch to", "hide", "launch", "bring up")
OS_FIXED = (
    "new tab", "close tab", "next tab", "previous tab", "reopen the closed tab", "new window",
    "close window", "minimize window", "full screen", "exit full screen", "show desktop",
    "mission control", "move window to the left", "move window to the right", "scroll down",
    "scroll up", "page down", "page up", "go back", "go forward", "scroll to the top",
    "scroll to the bottom", "zoom in", "zoom out", "select all", "copy that", "paste here", "undo",
    "redo", "delete that", "cut the line", "save the file", "save as", "print this", "volume up",
    "volume down", "mute", "unmute", "pause video", "play music", "next track", "previous track",
    "take a screenshot", "lock the screen", "turn on do not disturb", "turn off do not disturb",
    "brightness up", "brightness down", "turn off wifi", "turn on bluetooth", "empty the trash",
    "show notifications", "open spotlight", "what time is it", "start dictation", "stop listening",
    "go to sleep", "wake up", "show the clipboard", "open the downloads folder", "eject the disk",
    "turn on dark mode", "turn on night shift", "show battery level", "open a new finder window",
    "sort by date", "sort by name", "show hidden files", "duplicate this file", "move to trash",
    "get info", "quick look", "share this", "start screen recording", "stop screen recording")
FOLDER_NAMES = ("invoices", "holiday photos", "tax documents", "meeting notes", "design assets",
                "old projects", "receipts", "drafts", "client work", "screenshots")
REMINDERS = ("call the dentist", "buy milk", "send the invoice", "water the plants", "book the flight",
             "pick up the kids", "renew my passport", "pay the rent", "email the landlord", "take out the trash")
TIMES = ("noon", "five", "six thirty", "nine", "seven", "eight fifteen", "ten", "three", "two thirty", "midnight")
OS_TEMPLATES = (
    ("create a new folder called {x}", FOLDER_NAMES), ("rename this to {x}", FOLDER_NAMES),
    ("move this to {x}", FOLDER_NAMES), ("remind me to {x}", REMINDERS),
    ("set an alarm for {x}", TIMES), ("set a timer for {x} minutes", ("two", "five", "ten", "fifteen", "twenty", "thirty", "forty five")),
    ("set the volume to {x} percent", ("ten", "twenty", "thirty", "fifty", "seventy", "eighty", "one hundred")))
TOPICS = (
    "flights to tokyo", "the weather tomorrow", "noise cancelling headphones", "python list comprehension",
    "how to tie a tie", "restaurants near me", "the latest iphone", "chicken curry recipe",
    "basketball scores", "cheap hotels in lisbon", "standing desks", "how to center a div",
    "rust borrow checker", "used road bikes", "train times to boston", "the news today",
    "vegan dinner ideas", "marathon training plan", "react server components", "mortgage rates",
    "movie showtimes", "how to change a tire", "sourdough starter", "postgres index types",
    "best running shoes", "coffee shops open now", "kubernetes ingress", "apartment rentals",
    "electric cars", "piano lessons", "docker multi stage builds", "flu symptoms", "birthday gift ideas",
    "hiking trails nearby", "spanish verb conjugation", "typescript generics", "stock market today")
CLICK_VERBS = ("click", "click on", "press", "tap", "select", "double click", "choose", "hit")

BRANCHES = ("main", "develop", "staging", "release", "feature login", "hotfix", "the search branch", "production")
DIRS = ("source", "tests", "docs", "projects", "downloads", "the api folder", "the home directory", "scripts", "the build folder")
SCRIPTS = ("dev", "build", "test", "lint", "start", "deploy", "format", "watch")
HOSTS = ("the staging server", "production", "the build box", "the database host", "my raspberry pi")
PACKAGES = ("requests", "numpy", "react", "express", "pandas", "pytest", "typescript", "tailwind", "fast api", "lodash")
FILES = ("main dot pie", "package dot json", "the readme", "index dot js", "the config file",
         "the docker file", "the make file", "the env file", "app dot tsx", "settings dot pie")
NAMES = ("fix login", "new search", "dark mode", "user profile", "cleanup", "payment retry", "better errors", "upload limit")
TERMINAL_FIXED = (
    "git status", "git pull", "git push", "git diff", "git log", "git stash", "git stash pop",
    "git commit", "git add everything", "git fetch", "git reset hard", "git blame this file",
    "amend the last commit", "undo the last commit", "show the last five commits", "list files",
    "list all files", "clear the screen", "run the tests", "run the build", "run it again",
    "npm install", "npm start", "docker compose up", "docker compose down", "kill the process",
    "show running processes", "go up one directory", "print working directory",
    "open this folder in code", "install the dependencies", "restart the server", "stop the server",
    "check disk usage", "show memory usage", "who am i", "show my ip address", "exit the shell",
    "activate the virtual environment", "run the linter", "run the formatter", "watch the logs",
    "show environment variables", "cancel that", "run the migrations", "seed the database",
    "start the dev server", "build the docker image", "list docker containers", "pull the latest image",
    "show the git remotes", "clean the build folder", "run the last command again", "count the lines of code")
TERMINAL_TEMPLATES = (
    ("git checkout {x}", BRANCHES), ("git merge {x}", BRANCHES), ("git rebase onto {x}", BRANCHES),
    ("git push origin {x}", BRANCHES), ("git pull origin {x}", BRANCHES),
    ("switch to the {x} branch", BRANCHES), ("delete the {x} branch", BRANCHES),
    ("create a new branch called {x}", NAMES), ("commit with message {x}", NAMES),
    ("change directory to {x}", DIRS), ("list the files in {x}", DIRS), ("search {x} for to do", DIRS),
    ("npm run {x}", SCRIPTS), ("run the {x} script", SCRIPTS), ("ssh into {x}", HOSTS),
    ("copy the logs from {x}", HOSTS), ("install {x}", PACKAGES), ("uninstall {x}", PACKAGES),
    ("upgrade {x}", PACKAGES), ("open {x}", FILES), ("show me {x}", FILES), ("edit {x}", FILES),
    ("delete {x}", FILES), ("make a directory called {x}", NAMES), ("run {x}", FILES))

THINGS = (
    "this function", "this file", "this class", "this module", "the parser", "the login handler",
    "the user model", "the api client", "the cache layer", "the payment service", "the config loader",
    "the database schema", "the retry logic", "the auth middleware", "the search endpoint",
    "the upload handler", "the email sender", "the rate limiter", "the session store", "the test suite",
    "the build script", "the main loop", "the error handler", "the router", "the data pipeline",
    "the image resizer", "the token refresh", "the webhook handler", "the settings page",
    "the signup form", "the dashboard query", "the background worker")
DEV_TEMPLATES = (
    "refactor {x}", "explain {x}", "write a unit test for {x}", "write tests for {x}",
    "add type hints to {x}", "fix the bug in {x}", "add error handling to {x}", "add logging to {x}",
    "add a docstring to {x}", "document {x}", "optimize {x}", "simplify {x}", "rename {x}",
    "find where {x} is defined", "show me where {x} is used", "why is {x} so slow", "why does {x} fail",
    "make {x} async", "convert {x} to typescript", "rewrite {x} in rust", "review {x}", "summarize {x}",
    "clean up {x}", "split {x} into smaller functions", "add caching to {x}", "add retries to {x}",
    "write a migration for {x}", "mock {x} in the tests", "remove {x}", "what does {x} do",
    "add input validation to {x}", "is {x} thread safe", "add a timeout to {x}", "write a benchmark for {x}")
DEV_FIXED = (
    "why is this test failing", "explain this error", "generate a commit message",
    "review this pull request", "update the readme", "add a test for the edge case",
    "run the tests and fix any failures", "what changed in the last commit", "undo the last change",
    "apply the suggestion", "accept all changes", "reject the changes", "start a new chat", "continue",
    "try again", "make it shorter", "use a dictionary instead", "don't change the public api",
    "keep the existing behavior", "add a comment explaining why", "is there a simpler way",
    "what are the edge cases", "does this handle empty input", "fix the type errors",
    "fix the lint errors", "format the code", "remove the unused imports", "sort the imports",
    "bump the version", "open a pull request", "create a new file for this",
    "move this to a separate module", "revert this file", "where is this called from",
    "how do i run this locally", "what's the time complexity", "write the docs for this",
    "add an example to the readme", "stop generating", "that's not what i meant",
    "use the existing helper", "don't add new dependencies", "explain it step by step",
    "what would you do differently", "find the memory leak", "why is the build broken")

CHAT_THREADS = (
    ("are we still on for lunch", ("yes see you at noon", "sorry i can't make it", "can we push it to one", "running ten minutes late", "where do you want to eat")),
    ("what time will you be home", ("around six", "i'll be home by seven", "leaving the office now", "not sure yet i'll text you", "late tonight don't wait up")),
    ("did you get the invoice", ("yes got it thanks", "not yet can you resend it", "i'll pay it today", "it went to my spam folder")),
    ("can you review my pull request", ("sure give me ten minutes", "looking at it now", "left a few comments", "approved nice work", "can it wait until after standup")),
    ("happy birthday", ("thank you so much", "thanks that means a lot", "aw thank you")),
    ("the meeting moved to three", ("okay thanks for letting me know", "that works for me", "i have a conflict at three", "can we do four instead")),
    ("do you want to grab coffee", ("yes give me five minutes", "i'd love to", "maybe later i'm in a meeting", "sure where")),
    ("how was the interview", ("i think it went well", "really good actually", "tough but fair", "i'll tell you tonight")),
    ("are you coming to the party on saturday", ("yes i'll be there", "wouldn't miss it", "i can't i'm out of town", "what should i bring")),
    ("the deploy failed again", ("i'm looking into it", "roll it back", "what's the error", "did the migration run", "give me a minute")),
    ("can you send me the file", ("sending it now", "it's in the shared folder", "which one", "check your email")),
    ("i'm outside", ("coming down now", "be right there", "two minutes", "come on up")),
    ("did you feed the dog", ("yes this morning", "not yet i will now", "i thought you did")),
    ("thanks for your help today", ("no problem", "happy to help", "any time", "you're welcome")),
    ("should we order pizza", ("yes please", "sounds great", "i already ate", "get the pepperoni one")),
    ("are you free for a quick call", ("yes call me now", "give me five minutes", "can it wait until tomorrow", "i'm in a meeting until two")),
    ("i got the job", ("congratulations", "that's amazing news", "i knew you would", "we have to celebrate")),
    ("what do you think of the design", ("i love it", "the colors feel a bit loud", "can we try a darker version", "looks great ship it")),
    ("my flight is delayed", ("oh no how long", "that's annoying", "let me know when you land", "i'll pick you up whenever")),
    ("can you cover my shift on friday", ("sure no problem", "sorry i can't this week", "let me check my schedule")),
    ("is the report ready", ("almost done", "i'll send it in an hour", "it's on your desk", "i need one more day")),
    ("where are you", ("on my way", "almost there", "stuck in traffic", "i'm at the office", "five minutes away")),
    ("good morning", ("good morning", "morning how did you sleep", "hey good morning")),
    ("want to watch a movie tonight", ("yes what do you want to see", "i'm too tired tonight", "sure pick something funny")),
    ("the client wants the changes by monday", ("that's tight but doable", "we need more time", "i'll start on it today", "let's talk about it tomorrow")),
    ("i'm sorry about earlier", ("it's okay don't worry about it", "thank you for saying that", "let's talk when i get home")),
    ("can you pick up some milk", ("sure anything else", "already at the store", "i'll get it on my way home")),
    ("did the package arrive", ("yes it came this morning", "not yet", "it says out for delivery")),
)
CHAT_GENERIC = (
    "sounds good", "thanks so much", "i'll call you later", "no worries", "talk to you soon",
    "let me check and get back to you", "love you", "good night", "sorry i missed your call",
    "no thanks", "i agree", "great idea", "see you tomorrow", "see you soon", "let's do it",
    "that's so funny", "i'm so sorry to hear that", "let me know if you need anything",
    "have a great weekend", "can we talk later", "i'll think about it", "send me the link",
    "what's the address", "i'm on it", "will do", "got it thanks", "perfect see you then",
    "miss you", "drive safe", "text me when you get there", "i'm free after five", "not today sorry")
CHAT_PEOPLE = ("Sam", "Priya", "Mom", "Dad", "Alex", "Jordan", "Maria", "Chen", "Omar", "Lena", "Coach", "Team")

SITES = ("youtube", "github", "gmail", "hacker news", "wikipedia", "amazon", "reddit", "google maps",
         "linkedin", "netflix", "stack overflow", "the new york times")
BROWSER_FIXED = (
    "open a new tab", "bookmark this page", "open the first result", "open the second link",
    "add to cart", "play the video", "skip the ad", "translate this page", "reader mode",
    "download the file", "sign in", "log out", "refresh the page", "go back to the last page",
    "close the other tabs", "open in a private window", "copy the link", "find on this page",
    "scroll to the comments", "scroll to the reviews", "accept the cookies", "reject all cookies",
    "pin this tab", "mute this tab", "show my bookmarks", "show the history", "clear the search box",
    "open the last result", "pause the video", "make the video full screen", "turn on captions",
    "view the page source", "open the developer tools", "share this page", "save this as a pdf")

# (app, text of the focused element, nearby labels)
BROWSER_PAGES = (
    ("Sign in to your account", ("Email", "Password", "Sign in", "Forgot password", "Create account")),
    ("Shopping cart (3 items)", ("Checkout", "Remove", "Continue shopping", "Apply coupon", "Save for later")),
    ("Inbox (12) - Gmail", ("Compose", "Inbox", "Starred", "Sent", "Drafts", "Archive", "Reply", "Forward")),
    ("lofi beats to study to - YouTube", ("Subscribe", "Like", "Share", "Download", "Comments", "Skip ad")),
    ("Pull requests - GitHub", ("New pull request", "Files changed", "Merge", "Review changes", "Issues", "Actions")),
    ("Google Search results", ("Images", "News", "Videos", "Maps", "Shopping", "Tools")),
    ("Alan Turing - Wikipedia", ("Talk", "Edit", "View history", "Languages", "Contents")),
    ("Flights from San Francisco to New York", ("Sort by price", "Filters", "Nonstop only", "Select", "Book now")),
    ("Quarterly plan - Google Docs", ("Share", "Comment", "File", "Insert", "Format", "Suggest edits")),
    ("Accounts overview - Online banking", ("Transfer", "Pay bills", "Statements", "Deposit", "Log out")),
    ("Pricing - Plans for every team", ("Start free trial", "Contact sales", "Monthly", "Yearly", "Compare plans")),
    ("Hacker News", ("new", "past", "comments", "ask", "show", "jobs", "submit")),
    ("Checkout - Shipping address", ("Continue to payment", "Edit cart", "Use a different address", "Gift options")),
    ("Calendar - Week of the 12th", ("Create", "Today", "Week", "Month", "Settings")),
    ("Recipe: one pan chicken", ("Jump to recipe", "Print", "Save", "Reviews", "Nutrition")),
)
OS_SCREENS = (
    ("Finder", "Documents", ("Recents", "Applications", "Downloads", "Desktop", "AirDrop", "New Folder", "Get Info")),
    ("System Settings", "Wi-Fi", ("Bluetooth", "Network", "Sound", "Displays", "Focus", "Battery", "General")),
    ("Music", "Listen Now", ("Play", "Next", "Shuffle", "Library", "Radio", "Add to Playlist")),
    ("Photos", "Library - 4,212 Photos", ("Albums", "Favorites", "Edit", "Share", "Delete", "Import")),
    ("Mail", "Inbox - 8 unread", ("Reply", "Forward", "Archive", "Trash", "Flag", "New Message")),
    ("Calendar", "Today", ("Day", "Week", "Month", "Year", "New Event", "Calendars")),
    ("Notes", "Shopping list", ("New Note", "Pin Note", "Share", "Delete", "Folders", "Checklist")),
    ("Preview", "invoice-march.pdf", ("Markup", "Rotate", "Share", "Search", "Sidebar", "Zoom")),
    ("Finder", "Desktop", ("Documents", "Downloads", "Applications", "Trash", "New Folder")),
    ("Keynote", "Q3 review - slide 4", ("Play", "Add Slide", "Format", "Animate", "Share", "Comment")),
    ("Reminders", "Today - 3 reminders", ("New Reminder", "Scheduled", "Flagged", "Completed", "Lists")),
)
TERMINAL_APPS = ("Terminal", "iTerm2", "Ghostty", "Warp")
TERMINAL_TEXT = (
    "$ git status\nOn branch main\nYour branch is up to date with 'origin/main'.\nnothing to commit",
    "api git:(feature/login) x npm test\n 14 passing 2 failing",
    "npm ERR! missing script: start\n$",
    "FAILED tests/test_parser.py::test_tokens - AssertionError\n1 failed, 42 passed in 3.1s\n$",
    "$ docker ps\nCONTAINER ID  IMAGE        STATUS\n3f2a1c9d      api:latest   Up 2 hours\n$",
    "~/Projects/web main $ ls\nREADME.md  package.json  src  tests\n$",
    "error: failed to push some refs to 'origin'\nhint: Updates were rejected because the remote contains work\n$",
    "$ python manage.py runserver\nStarting development server at http://127.0.0.1:8000/",
    "CONFLICT (content): Merge conflict in src/app.ts\nAutomatic merge failed; fix conflicts\n$",
    "$ pytest -q\n..........F....\n$",
    "Last login: Sat Nov 21 09:14:02 on ttys003\n~ $",
    "$ make build\ngcc -O2 -o bin/server src/main.c\n$",
)
TERMINAL_LABELS = (("zsh", "main", "~/Projects/api"), ("bash", "develop", "~/code/web"),
                   ("zsh", "feature/login", "tmux"), ("fish", "main", "ssh build-box"), ("zsh", "Split Pane", "New Tab"))
DEV_APPS = ("Cursor", "Visual Studio Code", "Claude", "Xcode", "Zed", "ChatGPT")
DEV_TEXT = (
    "def parse_tokens(source):\n    tokens = []\n    for line in source.splitlines():",
    "async function fetchUser(id) {\n  const res = await fetch(`/api/users/${id}`);",
    "class PaymentService:\n    def charge(self, user, amount):",
    "TypeError: Cannot read properties of undefined (reading 'map')\n    at renderList (list.tsx:42)",
    "Ask anything about your code",
    "func loadSettings() throws -> Settings {\n    let data = try Data(contentsOf: url)",
    "FAILED test_login_redirects - assert 302 == 200",
    "SELECT id, email FROM users WHERE last_login < now() - interval '90 days'",
    "export default function Dashboard({ user }) {\n  const [rows, setRows] = useState([])",
    "Message Claude",
    "impl Cache {\n    pub fn get(&self, key: &str) -> Option<&Entry> {",
    "// TODO: retry with backoff when the upstream returns 503",
)
DEV_LABELS = (("Explorer", "Run and Debug", "main.py", "Problems", "Terminal"), ("Chat", "Apply", "Accept", "Reject", "client.ts"),
              ("Source Control", "Extensions", "Outline", "Timeline"), ("New chat", "Attach file", "Send", "Model"),
              ("Build", "Run", "Debug Navigator", "Breakpoints"))
CHAT_APPS = ("Messages", "Slack", "WhatsApp", "Discord", "Telegram", "Signal")
CHAT_LABELS = (("iMessage", "Details", "FaceTime"), ("Threads", "Huddle", "Mentions", "#general"),
               ("Chats", "Calls", "Status", "Attach"), ("Send", "Emoji", "Voice message"))
BROWSER_APPS = ("Safari", "Google Chrome", "Arc", "Firefox")


def _click_target(labels, rng):
    label = rng.choice(labels)
    return f"{rng.choice(CLICK_VERBS)} {label.lower()}"


def sample_os(rng):
    app, text, labels = rng.choice(OS_SCREENS)
    context = dict(app_name=app, ax_role=rng.choice(("AXWindow", "AXButton", "AXStaticText", "AXGroup", "AXMenuItem")),
                   ax_text=text, nearby_labels=list(labels))
    roll = rng.random()
    if roll < 0.22:
        return f"{rng.choice(APP_VERBS)} {rng.choice(APPS)}", context
    if roll < 0.40:
        context["ax_role"] = "AXButton"
        return _click_target(labels, rng), context
    if roll < 0.55:
        template, values = rng.choice(OS_TEMPLATES)
        return template.format(x=rng.choice(values)), context
    return rng.choice(OS_FIXED), context


def sample_terminal(rng):
    context = dict(app_name=rng.choice(TERMINAL_APPS), ax_role="AXTextArea", ax_text=rng.choice(TERMINAL_TEXT),
                   nearby_labels=list(rng.choice(TERMINAL_LABELS)))
    if rng.random() < 0.5:
        template, values = rng.choice(TERMINAL_TEMPLATES)
        return template.format(x=rng.choice(values)), context
    return rng.choice(TERMINAL_FIXED), context


def sample_dev(rng):
    context = dict(app_name=rng.choice(DEV_APPS), ax_role=rng.choice(("AXTextArea", "AXTextField")),
                   ax_text=rng.choice(DEV_TEXT), nearby_labels=list(rng.choice(DEV_LABELS)))
    if rng.random() < 0.6:
        return rng.choice(DEV_TEMPLATES).format(x=rng.choice(THINGS)), context
    return rng.choice(DEV_FIXED), context


def sample_chat(rng):
    person = rng.choice(CHAT_PEOPLE)
    incoming, replies = rng.choice(CHAT_THREADS)
    reply = rng.choice(replies) if rng.random() < 0.7 else rng.choice(CHAT_GENERIC)
    context = dict(app_name=rng.choice(CHAT_APPS), ax_role=rng.choice(("AXTextArea", "AXTextField")),
                   ax_text=f"{person}: {incoming}", nearby_labels=[person] + list(rng.choice(CHAT_LABELS)))
    return reply, context


def sample_browser(rng):
    title, labels = rng.choice(BROWSER_PAGES)
    context = dict(app_name=rng.choice(BROWSER_APPS), ax_role=rng.choice(("AXWebArea", "AXLink", "AXButton", "AXTextField")),
                   ax_text=title, nearby_labels=list(labels))
    roll = rng.random()
    if roll < 0.30:
        context["ax_role"] = rng.choice(("AXButton", "AXLink"))
        return _click_target(labels, rng), context
    if roll < 0.50:
        context["ax_role"] = "AXTextField"
        return f"{rng.choice(('search for', 'look up', 'google'))} {rng.choice(TOPICS)}", context
    if roll < 0.62:
        return f"{rng.choice(('go to', 'open'))} {rng.choice(SITES)}", context
    return rng.choice(BROWSER_FIXED), context


ANY_APPS = tuple(sorted({"Finder", "Safari", "Mail", "Notes", "Pages", "Preview", "TextEdit", "Google Chrome"}
                        | set(TERMINAL_APPS) | set(DEV_APPS) | set(CHAT_APPS)))
ANY_ROLES = ("AXTextArea", "AXTextField", "AXStaticText", "AXWebArea", "AXButton", "AXLink", "AXGroup")


def sample_vocab(rng):
    """General-vocabulary drill: a few common words, so the model learns to read phonemes it has
    never seen in a command. Half the time some of the words are visible on screen among
    distractors, which is the only thing that can settle a homophene (pat / bat / mat)."""
    words, cumulative = common_words()
    count = rng.choices((1, 2, 3, 4, 5, 6), (15, 25, 25, 20, 10, 5))[0]
    target = rng.choices(words, cum_weights=cumulative, k=count)
    roll = rng.random()
    if roll < 0.25:
        return " ".join(target), dict(app_name="", ax_role="", ax_text="", nearby_labels=[])
    pool = rng.choices(words, cum_weights=cumulative, k=rng.randint(3, 7))
    if roll < 0.75:
        pool += [w for w in target if rng.random() < 0.7] or [target[0]]
    rng.shuffle(pool)
    cut = rng.randint(0, min(4, len(pool) - 1))
    return " ".join(target), dict(app_name=rng.choice(ANY_APPS), ax_role=rng.choice(ANY_ROLES),
                                  ax_text=" ".join(pool[cut:]), nearby_labels=[w.capitalize() for w in pool[:cut]])


DOMAINS = dict(os=sample_os, terminal=sample_terminal, dev=sample_dev, chat=sample_chat, browser=sample_browser,
               vocab=sample_vocab)
DEFAULT_WEIGHTS = dict(os=0.10, terminal=0.10, dev=0.12, chat=0.10, browser=0.08, vocab=0.50)

# ---------------------------------------------------------------------------------------------
# Homophene challenge: two phrases with the same viseme sequence, each natural in a different app.
# The lattice is identical for both, so only the screen context can decide. Never used for training.
# ---------------------------------------------------------------------------------------------
def _ctx(app, role, text, *labels):
    return dict(app_name=app, ax_role=role, ax_text=text, nearby_labels=list(labels))


CHALLENGE = (
    ("restart the pod", _ctx("Terminal", "AXTextArea", "$ kubectl get pods\nNAME      READY  STATUS            RESTARTS\napi-7d9f  0/1    CrashLoopBackOff  12\n$", "zsh", "kubectl", "prod-cluster"),
     "restart the bot", _ctx("Discord", "AXTextArea", "Kai: ModBot stopped answering commands in the server again", "#general", "ModBot", "Server Settings", "Members")),
    ("fix the sync", _ctx("Visual Studio Code", "AXTextArea", "// TODO: calendar sync drops events after a token refresh\nasync function syncCalendar(account) {", "sync.ts", "Problems", "Source Control"),
     "fix the sink", _ctx("Messages", "AXTextArea", "Landlord: the plumber can come Thursday, what needs doing in the kitchen?", "Landlord", "iMessage", "Details")),
    ("check the cache", _ctx("Terminal", "AXTextArea", "$ redis-cli info memory\nused_memory_human:1.52G\nevicted_keys:48213\n$", "zsh", "redis-cli", "api-server"),
     "check the cash", _ctx("Numbers", "AXCell", "Till count Friday: bills 380, coins 32.50, expected 430", "Cash drawer", "Deposits", "Sheet 1")),
    ("check the route", _ctx("Maps", "AXTextField", "Directions to the airport: 42 min via US-101 S, heavy traffic ahead", "Directions", "Drive", "Leave now", "Avoid tolls"),
     "check the root", _ctx("Terminal", "AXTextArea", "$ df -h /\nFilesystem     Size  Used  Capacity  Mounted on\n/dev/disk3s1  926Gi  901Gi  98%  /\n$", "zsh", "sudo", "ssh build-box")),
    ("delete the node", _ctx("Terminal", "AXTextArea", "$ kubectl get nodes\nNAME      STATUS    ROLES\nworker-3  NotReady  <none>\n$", "zsh", "kubectl", "staging"),
     "delete the note", _ctx("Notes", "AXTextArea", "Grocery list: milk, eggs, coffee (old, already bought)", "All iCloud", "Pin Note", "Share", "New Folder")),
    ("push the tag", _ctx("Terminal", "AXTextArea", "$ git tag v2.4.0\n$ git tag\nv2.3.9\nv2.4.0\n$", "zsh", "main", "~/Projects/api"),
     "push the dag", _ctx("Google Chrome", "AXWebArea", "Airflow - DAGs - daily_sales_etl - edited locally, not yet deployed", "DAGs", "Trigger DAG", "Code", "Graph")),
    ("show me the code", _ctx("Cursor", "AXTextArea", "The retry logic lives in src/http/client.ts, inside the withBackoff helper.", "Chat", "client.ts", "Apply", "Explorer"),
     "show me the coat", _ctx("Safari", "AXWebArea", "Wool Blend Overcoat - Camel - $189 - Free returns", "Add to Bag", "Size guide", "Coats & Jackets", "Reviews")),
    ("fix the path", _ctx("Terminal", "AXTextArea", "zsh: command not found: poetry\n$ echo $PATH\n/usr/bin:/bin:/usr/sbin\n$", "zsh", "~/.zshrc", "main"),
     "fix the math", _ctx("Numbers", "AXCell", "=SUM(B2:B9)*0.8 shows 1,240 but the total should be 1,550", "Formula", "Sheet 1", "Budget", "Sum")),
    ("open the port", _ctx("Terminal", "AXTextArea", "$ curl localhost:8080\ncurl: (7) Failed to connect to localhost port 8080: Connection refused\n$ sudo ufw status", "zsh", "firewall", "ssh"),
     "open the board", _ctx("Google Chrome", "AXWebArea", "Trello - Workspaces - Sprint 14 board - Roadmap board", "Boards", "Sprint 14", "Create", "Templates")),
    ("just one byte", _ctx("Visual Studio Code", "AXTextArea", "uint8_t buf[1];\nssize_t n = read(fd, buf, sizeof(buf)); // how much should we read per call?", "serial.c", "Debug Console", "Run"),
     "just one bite", _ctx("Messages", "AXTextArea", "Sam: you HAVE to try this cake, how much do you want?", "Sam", "iMessage", "Details")),
    ("merge it", _ctx("Google Chrome", "AXWebArea", "Pull request #482 - All checks have passed - 2 approvals - This branch has no conflicts", "Merge pull request", "Files changed", "Conversation", "Squash and merge"),
     "purge it", _ctx("Terminal", "AXTextArea", "$ docker system df\nBuild Cache   41.2GB   reclaimable\n# the CDN is still serving the stale bundle\n$", "zsh", "docker", "cloudflare")),
    ("start the batch", _ctx("Terminal", "AXTextArea", "$ ls jobs/\nnightly_batch.sh  README.md\n$ squeue\nJOBID  PARTITION  NAME  ST\n$", "zsh", "slurm", "hpc-login"),
     "start the match", _ctx("Discord", "AXTextArea", "Lobby is full, everyone is ready, waiting for the host", "#scrims", "Voice Connected", "Ready")),
    ("back it up", _ctx("Terminal", "AXTextArea", "$ ls ~/Documents/thesis\nchapter1.tex  chapter2.tex  figures/\n# this laptop has the only copy\n$", "zsh", "rsync", "Time Machine"),
     "pack it up", _ctx("Messages", "AXTextArea", "Jo: movers get here at 8 tomorrow and the kitchen stuff is still everywhere", "Jo", "iMessage", "Details")),
    ("open the tab", _ctx("Safari", "AXWebArea", "Recently closed: Hacker News, Gmail, Flight to Lisbon", "History", "Recently Closed", "Tab Overview", "Bookmarks"),
     "open the tap", _ctx("Messages", "AXTextArea", "Dad: hose is connected to the garden faucet, ready when you are", "Dad", "iMessage", "Details")),
    ("find the bug", _ctx("Xcode", "AXTextArea", "Thread 1: Fatal error: Index out of range\n    let item = items[index]", "Debug Navigator", "Breakpoints", "Console"),
     "find the mug", _ctx("Safari", "AXWebArea", "Search results for ceramic: 48 items - Stoneware Mug $14 - Espresso Cup $9", "Mugs", "Kitchen", "Add to Cart", "Sort by")),
    ("show the dock", _ctx("System Settings", "AXCheckBox", "Desktop & Dock - Automatically hide and show the Dock: On", "Desktop & Dock", "Magnification", "Position on screen"),
     "show the doc", _ctx("Slack", "AXTextArea", "Priya: I finished the launch plan doc, want to see it before standup?", "#launch", "Priya", "Threads", "Huddle")),
    ("the beat is great", _ctx("GarageBand", "AXGroup", "Track 3 - Drum Machine - 92 BPM - loop region 1 to 8", "Drummer", "Tempo", "Loop Browser"),
     "the meat is great", _ctx("Messages", "AXTextArea", "Ana: how's the brisket at that new barbecue place?", "Ana", "iMessage", "Details")),
    ("add the mat to the cart", _ctx("Safari", "AXWebArea", "Yoga & Pilates - Non-Slip 6mm Mat - $32 - In stock", "Add to Cart", "Yoga", "Blocks & Straps"),
     "add the bat to the cart", _ctx("Safari", "AXWebArea", "Baseball - Maple Wood Bat 33in - $79 - In stock", "Add to Cart", "Baseball", "Gloves", "Helmets")),
    ("send the deck", _ctx("Slack", "AXTextArea", "Maria: can I get the investor slides before the 3pm call?", "#fundraising", "Maria", "Keynote"),
     "send the tech", _ctx("Messages", "AXTextArea", "IT Support: ticket 4412, the printer on floor 2 is still offline. How do you want to proceed?", "IT Support", "iMessage", "Details")),
    ("add the crate", _ctx("Terminal", "AXTextArea", "error[E0432]: unresolved import `serde`\n --> src/main.rs:1:5\n$", "zsh", "cargo", "Cargo.toml"),
     "add the grade", _ctx("Numbers", "AXCell", "Gradebook - Student: L. Chen - Midterm: 88 - Final: (empty)", "Grades", "Final", "Sheet 1")),
)
CHALLENGE_PHRASES = frozenset(p for a, _, b, _ in CHALLENGE for p in (a, b))

# ---------------------------------------------------------------------------------------------
# Lattices
# ---------------------------------------------------------------------------------------------
def flat_phonemes(text):
    """Stress-marked ARPAbet for a phrase, words concatenated (lips show no word boundaries)."""
    return [p for _, phones in g2p(text) for p in phones]


def _posterior(base, rng, severity):
    """Sampled lip-reader posterior for one true phoneme: {phoneme: probability}."""
    members = homophenes(base)
    if base not in VOWELS and len(members) > 1:
        draws = {p: rng.gammavariate(1.2, 1.0) for p in members}  # near-uniform inside the class
        total = sum(draws.values())
        weights = {p: 0.93 * d / total for p, d in draws.items()}
    else:
        main = min(0.93, max(0.35, rng.betavariate(8, 2) - 0.08 * (severity - 1)))  # leaves mass for look-alikes
        weights = {base: main}
        others = [p for p in members if p != base] + list(NEAR.get(base, ()))
        if others:
            picks = rng.sample(others, min(len(others), rng.choice((1, 1, 2))))
            share = (0.95 - main) / len(picks)
            for p in picks:
                weights[p] = weights.get(p, 0.0) + share
            if rng.random() < 0.15 * severity:  # the reader prefers the wrong look-alike
                wrong = picks[0]
                weights[base], weights[wrong] = weights[wrong], weights[base]
    stray = rng.choice([p for p in PHONEMES if p not in weights])
    weights[stray] = max(1.0 - sum(weights.values()), 0.01)
    top = max(weights, key=weights.get)
    for p in homophenes(top):  # lattice contract: the top phoneme's whole class is listed
        weights.setdefault(p, 0.01)
    total = sum(weights.values())
    return {p: w / total for p, w in weights.items()}


def _uniform(base):
    """No preference inside the viseme class. Tiny decreasing offsets keep the class order stable
    without favouring the true phoneme."""
    members = VISEME_CLASSES[VISEME_OF[base]]
    return {p: 1.0 / len(members) - 1e-6 * i for i, p in enumerate(members)}


def _assemble(posteriors, durations, gaps, lead, tail):
    """Slots + top-K hypotheses + frame spans from per-slot posteriors."""
    slots, t = [], lead
    for weights, duration, gap in zip(posteriors, durations, gaps):
        ranked = sorted(weights.items(), key=lambda kv: -kv[1])
        slots.append(dict(frames=[t, t + duration - 1],
                          alternatives=[[p, round(math.log(w), 4)] for p, w in ranked]))
        t += duration + gap
    frames = [s["frames"] for s in slots]
    best = [s["alternatives"][0] for s in slots]
    best_score = sum(lp for _, lp in best)
    candidates = [([p for p, _ in best], best_score)]
    margins = sorted((s["alternatives"][0][1] - s["alternatives"][1][1], i)
                     for i, s in enumerate(slots) if len(s["alternatives"]) > 1)
    for margin, i in margins[:TOP_K - 1]:
        phones = [p for p, _ in best]
        phones[i] = slots[i]["alternatives"][1][0]
        candidates.append((phones, best_score - margin))
    hypotheses = [dict(phonemes=phones, log_prob=round(score, 4), frames=frames)
                  for phones, score in sorted(candidates, key=lambda c: -c[1])[:TOP_K]]
    return dict(frame_rate=FRAME_RATE, num_frames=t + tail, hypotheses=hypotheses, slots=slots)


def noisy_lattice(text, rng, severity=1.0, ambiguity="sampled"):
    """A lip-reading-like lattice for `text`. `severity` scales deletions, swaps and insertions.
    ambiguity="sampled" draws a posterior per sound; "uniform" gives no preference inside a class."""
    draw = (lambda base: _uniform(base)) if ambiguity == "uniform" else (lambda base: _posterior(base, rng, severity))
    posteriors = []
    for marked in flat_phonemes(text):
        base = strip_stress(marked)
        if base in VOWELS:
            p_delete = 0.28 if marked.endswith("0") else 0.03
        else:
            p_delete = 0.35 if base == "HH" else 0.04
        if rng.random() < min(0.9, p_delete * severity):
            continue
        posteriors.append(draw(base))
        if rng.random() < 0.02 * severity:
            posteriors.append(draw(rng.choice(("AH", "AH", "IH", rng.choice(PHONEMES)))))
    if not posteriors:
        posteriors.append(draw(strip_stress(flat_phonemes(text)[0])))
    durations = [rng.randint(1, 4) + (max(w, key=w.get) in VOWELS) for w in posteriors]
    gaps = [rng.randint(0, 2) for _ in posteriors]
    return _assemble(posteriors, durations, gaps, rng.randint(3, 8), rng.randint(3, 8))


def ambiguous_lattice(text):
    """Deterministic lattice that is uniform over each viseme class, in class order. Two phrases
    with the same viseme sequence get byte-identical lattices."""
    posteriors = [_uniform(strip_stress(marked)) for marked in flat_phonemes(text)]
    n = len(posteriors)
    return _assemble(posteriors, [3] * n, [1] * n, 4, 4)


# ---------------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------------
def split_of(text):
    """Stable split by phrase, so no test phrase is ever trained on."""
    bucket = int(sha1(normalize_text(text).encode()).hexdigest(), 16) % 100
    return "valid" if bucket < 6 else "test" if bucket < 16 else "train"


def generate(n, seed=0, max_per_phrase=4, weights=None, uniform_share=0.3):
    """n records: dict(text, domain, split, screen_context, lattice). Deterministic for a seed.
    A phrase is reused at most `max_per_phrase` times, each time with new noise and context.
    `uniform_share` of the lattices carry no preference inside a viseme class."""
    rng = random.Random(seed)
    weights = weights or DEFAULT_WEIGHTS
    names = sorted(weights)
    counts, records, attempts = {}, [], 0
    while len(records) < n:
        attempts += 1
        if attempts > 200 * n:
            raise RuntimeError(f"Only {len(records)} examples available under max_per_phrase={max_per_phrase}")
        domain = rng.choices(names, [weights[k] for k in names])[0]
        text, context = DOMAINS[domain](rng)
        text = normalize_text(text)
        if not text or text in CHALLENGE_PHRASES or counts.get(text, 0) >= max_per_phrase:
            continue
        counts[text] = counts.get(text, 0) + 1
        ambiguity = "uniform" if rng.random() < uniform_share else "sampled"
        records.append(dict(text=text, domain=domain, split=split_of(text), screen_context=context,
                            lattice=noisy_lattice(text, rng, rng.uniform(0.5, 1.5), ambiguity)))
    return records


def challenge_records():
    """Two records per pair: the same lattice, a different screen context, a different answer."""
    records = []
    for pair, (text_a, context_a, text_b, context_b) in enumerate(CHALLENGE):
        lattice = ambiguous_lattice(text_a)
        for text, context in ((text_a, context_a), (text_b, context_b)):
            records.append(dict(text=text, domain="challenge", split="challenge", pair=pair,
                                screen_context=context, lattice=lattice))
    return records


def write_jsonl(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_dataset(out, n, seed=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = generate(n, seed)
    sizes = {}
    for split in ("train", "valid", "test"):
        rows = [r for r in records if r["split"] == split]
        write_jsonl(out / f"{split}.jsonl", rows)
        sizes[split] = len(rows)
    write_jsonl(out / "challenge.jsonl", challenge_records())
    sizes["challenge"] = 2 * len(CHALLENGE)
    sizes["unique_phrases"] = len({r["text"] for r in records})
    return sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(json.dumps(write_dataset(args.out, args.n, args.seed)))


if __name__ == "__main__":
    main()
