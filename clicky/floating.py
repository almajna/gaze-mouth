"""Show a small floating button above the Dock when the menu bar has no room for CursorMix's icon."""

import AppKit as A

SIZE = 44


def icon_hidden(status_item):
    """Report whether macOS left the status item out of the visible menu bar."""
    window = status_item.button().window()
    screen = window.screen() if window else None
    if screen is None:
        return True
    beside_notch = screen.auxiliaryTopRightArea()
    if beside_notch.size.width > 0:
        # Items that do not fit beside the notch are parked behind it, left of the visible area.
        return window.frame().origin.x < beside_notch.origin.x
    return not window.occlusionState() & A.NSWindowOcclusionStateVisible


class FloatingButton:
    def __init__(self, delegate):
        self.window = A.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (SIZE, SIZE)),
            A.NSWindowStyleMaskBorderless | A.NSWindowStyleMaskNonactivatingPanel,
            A.NSBackingStoreBuffered,
            False,
        )
        self.window.setLevel_(A.NSStatusWindowLevel)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(A.NSColor.clearColor())
        self.window.setHasShadow_(True)
        self.window.setHidesOnDeactivate_(False)
        self.window.setReleasedWhenClosed_(False)
        self.window.setCollectionBehavior_(
            A.NSWindowCollectionBehaviorCanJoinAllSpaces
            | A.NSWindowCollectionBehaviorStationary
            | A.NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        button = A.NSButton.alloc().initWithFrame_(((0, 0), (SIZE, SIZE)))
        button.setBordered_(False)
        button.setWantsLayer_(True)
        button.layer().setCornerRadius_(SIZE / 2)
        button.layer().setBackgroundColor_(A.NSColor.systemOrangeColor().CGColor())
        icon = A.NSImage.imageWithSystemSymbolName_accessibilityDescription_("cursorarrow.rays", "CursorMix")
        button.setImage_(icon.imageWithSymbolConfiguration_(
            A.NSImageSymbolConfiguration.configurationWithPointSize_weight_(19, A.NSFontWeightSemibold)
        ))
        button.setContentTintColor_(A.NSColor.whiteColor())
        button.setToolTip_("CursorMix")
        button.setAccessibilityLabel_("CursorMix menu")
        button.setTarget_(delegate)
        button.setAction_("showFloatingMenu:")
        self.window.setContentView_(button)

    def sync(self, hidden):
        """Show the button while the menu-bar icon is hidden, at the right end of the Dock."""
        if not hidden:
            self.window.orderOut_(None)
            return
        screen = A.NSScreen.screens()[0]
        visible = screen.visibleFrame()
        self.window.setFrameOrigin_((A.NSMaxX(visible) - SIZE - 14, A.NSMinY(visible) + 14))
        self.window.orderFrontRegardless()

    def close(self):
        self.window.orderOut_(None)
