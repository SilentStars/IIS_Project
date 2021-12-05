#%%
import importlib
import AudioPlayer
importlib.reload(AudioPlayer)

player = AudioPlayer.AudioPlayer()
# %%
furhat = player.furhat
# %%
blink_left={
    "frames": [
        {
            "time": [
                0.33
            ],
            "params": {
                "BLINK_LEFT": 2.0
            }
        },
        {
            "time": [
                1.0
            ],
            "params": {
                "reset": True
            }
        }
    ],
    "class": "furhatos.gestures.Gesture"
}
# %%
furhat.gesture(definition={
    "frames": [
        {
            "time": [
                0.33
            ],
            "params": {
                "BLINK_LEFT": 2.0
            }
        },
        {
            "time": [
                0.67
            ],
            "params": {
                "reset": True
            }
        }
    ],
    "class": "furhatos.gestures.Gesture"
    })
# %%
del(player)
# %%
gesture={
    "frames": [
        {
            "time": [
                0.33
            ],
            "params": {
                "BLINK_LEFT": 1.0,
                "persist" : True,
                "priority":1000
            },
            'priority':1000
        }
    ],
    "class": "furhatos.gestures.Gesture",
    "priority":1000
}
# %%
