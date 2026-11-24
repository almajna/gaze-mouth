"""Register only the chosen key combination with the macOS event dispatcher."""

import ctypes as C

SHORTCUTS = {
    "Control Option Space": (49, (1 << 12) | (1 << 11)),
    "Option Space": (49, 1 << 11),
    "Command Shift Space": (49, (1 << 8) | (1 << 9)),
}
DICTATION_SHORTCUTS = {
    "Control Option D": (2, (1 << 12) | (1 << 11)),
    "Control Shift D": (2, (1 << 12) | (1 << 9)),
}

EVENT_HOT_KEY_EXCLUSIVE = 1 << 0  # kEventHotKeyExclusive in CarbonEvents.h
EVENT_HOT_KEY_EXISTS_ERR = -9878


def shortcut_label(name):
    symbols = {"Control": "⌃", "Option": "⌥", "Command": "⌘", "Shift": "⇧"}
    glyphs = "".join(symbols.get(key, key) for key in name.split())
    return f"{name.replace(' ', '–')} ({glyphs})"


class EventType(C.Structure):
    _fields_ = [("event_class", C.c_uint32), ("event_kind", C.c_uint32)]


class HotKeyID(C.Structure):
    _fields_ = [("signature", C.c_uint32), ("identifier", C.c_uint32)]


Handler = C.CFUNCTYPE(C.c_int32, C.c_void_p, C.c_void_p, C.c_void_p)


class Hotkey:
    def __init__(self, on_press, on_release=None, *, identifier=1, choices=SHORTCUTS):
        self.down = False
        self.identifier = identifier
        self.choices = choices
        self.carbon = C.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
        self.registration = C.c_void_p()
        self.handler = C.c_void_p()
        self.shortcut = None
        self.carbon.GetApplicationEventTarget.restype = C.c_void_p
        self.target = self.carbon.GetApplicationEventTarget()
        self.carbon.InstallEventHandler.argtypes = [
            C.c_void_p,
            Handler,
            C.c_uint32,
            C.POINTER(EventType),
            C.c_void_p,
            C.POINTER(C.c_void_p),
        ]
        self.carbon.RegisterEventHotKey.argtypes = [
            C.c_uint32,
            C.c_uint32,
            HotKeyID,
            C.c_void_p,
            C.c_uint32,
            C.POINTER(C.c_void_p),
        ]
        self.carbon.UnregisterEventHotKey.argtypes = [C.c_void_p]
        self.carbon.RemoveEventHandler.argtypes = [C.c_void_p]

        self.carbon.GetEventKind.argtypes = [C.c_void_p]
        self.carbon.GetEventKind.restype = C.c_uint32
        self.carbon.GetEventParameter.argtypes = [
            C.c_void_p,
            C.c_uint32,
            C.c_uint32,
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.c_void_p,
        ]

        def pressed(call, event, data):
            key = HotKeyID()
            status = self.carbon.GetEventParameter(
                event,
                int.from_bytes(b"----"),
                int.from_bytes(b"hkid"),
                None,
                C.sizeof(key),
                None,
                C.byref(key),
            )
            if status or key.signature != int.from_bytes(b"Clky") or key.identifier != self.identifier:
                return -9874  # eventNotHandledErr passes other registrations to their own handlers.
            if self.carbon.GetEventKind(event) == 6:
                if not self.down:
                    return 0
                self.down = False
                if on_release:
                    on_release()
            elif not self.down:
                self.down = True
                on_press()
            return 0

        self.callback = Handler(pressed)
        keyboard = int.from_bytes(b"keyb")
        events = (EventType * 2)(EventType(keyboard, 5), EventType(keyboard, 6))
        status = self.carbon.InstallEventHandler(
            self.target, self.callback, 2, events, None, C.byref(self.handler)
        )
        if status:
            raise RuntimeError(f"macOS could not install the shortcut handler ({status}).")

    def register(self, name):
        if self.shortcut == name:
            return
        code, modifiers = self.choices[name]
        registration = C.c_void_p()
        status = self.carbon.RegisterEventHotKey(
            code,
            modifiers,
            HotKeyID(int.from_bytes(b"Clky"), self.identifier),
            self.target,
            EVENT_HOT_KEY_EXCLUSIVE,
            C.byref(registration),
        )
        if status == EVENT_HOT_KEY_EXISTS_ERR:
            raise RuntimeError(f"{shortcut_label(name)} is already reserved by another shortcut.")
        if status:
            raise RuntimeError(f"macOS could not register {name} ({status}). Choose another shortcut.")
        if self.registration:
            self.carbon.UnregisterEventHotKey(self.registration)
        self.registration = registration
        self.shortcut = name
        self.down = False

    def close(self):
        self.down = False
        if self.registration:
            self.carbon.UnregisterEventHotKey(self.registration)
            self.registration = C.c_void_p()
            self.shortcut = None
        if self.handler:
            self.carbon.RemoveEventHandler(self.handler)
            self.handler = C.c_void_p()
