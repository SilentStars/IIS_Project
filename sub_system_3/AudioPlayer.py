import os
from furhat_remote_api import FurhatRemoteAPI
from glob import glob


class AudioPlayer():

    def __init__(self,src_repo="src",port="8000") -> None:
        self.src_path = f"http://localhost:{port}/{src_repo}"
        self.furhat = FurhatRemoteAPI("localhost")

        self.music_list = list(map(os.path.basename,glob(os.path.join(src_repo,'*.wav'))))
        self.nb_music = len(self.music_list)
        self.index = 0
        self.index_favorite = None
        self.music_playing = False

        # Start the server
        os.system('./serve_file.sh')
        print('server is up')
        with open('pid','r') as f:
            self.pid = f.read()
        print(f"process running on {self.pid}")

    def _increase_index(self)->None:
        """Increases the index of the current music by one.
        """
        self.index = (self.index + 1) % self.nb_music

    def _decrease_index(self)->None:
        """Decreases the index of the current music by one
        """
        self.index = (self.index - 1) % self.nb_music

    def play(self)->None:
        """Robots start playing the current indexed song
        """
        _url = self._get_url()
        self.furhat.say(url=_url,lipsync=True)
        print(f"{self.music_list[self.index]} is playing")
        self.music_playing = True

    def play_favorite(self)->None:
        if self.index_favorite is not None:
            _url = self._get_url(index=self.index_favorite)
            print("play favorite song")
            if self.music_playing:
                self.stop()
            self.furhat.say(url=_url,lipsync=True)
            print("favorite song playing")
            self.music_playing = True

    def add_favorite(self)->None:
        self.index_favorite = self.index
        print(f"favorite music has been set to {self.music_list[self.index]}")

    def stop(self)->None:
        """Stops the robot from playing music
        """
        self.furhat.say_stop()
        self.music_playing = False
        print(f"music is stopped")

    def _get_file_path(self,music_item:str)->str:
        """Generates the url for a file from `self.music_list`

        Args:
            music_item (str): item of `self.music_list`

        Returns:
            str: url
        """
        return f"{self.src_path}/{music_item}"

    def _get_url(self,index=None)->str:
        """Returns url of the current indexed song

        Returns:
            str: url
        """
        if index is None : index = self.index
        return self._get_file_path(self.music_list[index])
    
    def next(self)->None:
        """Handle gesture C
        If a music is currently playing, we start play the following
        Otherwise we only increase the index

        Args:
            music_playing (bool, optional): Wheter or not a music is currently played. Defaults to False.
        """
        print("switch to next song")
        self._increase_index()
        if self.music_playing:
            self.stop()
            self.play()

    def previous(self)->None:
        """Handle gesture L
        If a music is currently playing, we start play the previous
        Otherwise we only decrease the index

        Args:
            music_playing (bool, optional): [description]. Defaults to False.
        """
        print("go back to previous song")
        self._decrease_index()
        if self.music_playing:
            self.stop()
            self.play()
            

    def __del__(self):
        os.system(f'kill {self.pid}')
        print('server has been killed')

        

if __name__ == "__main__":
    player = AudioPlayer()

