#%%
import importlib
import AudioPlayer
importlib.reload(AudioPlayer)

player = AudioPlayer.AudioPlayer()
# %%
player.index
# %%
player.next(music_playing=False)
#%%
player.next(music_playing=True)
# %%
del(player)

# %%
