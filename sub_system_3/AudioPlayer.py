import os
from furhat_remote_api import FurhatRemoteAPI
from glob import glob


class AudioPlayer():

    def __init__(self,src_repo="src",port="8000") -> None:
        self.src_path = f"http://localhost:{port}/{src_repo}"
        self.furhat = FurhatRemoteAPI("localhost")

        self.music_list = os.listdir(src_repo)
        self.nb_music = len(self.music_list)
        self.index = 0

        # Start the server
        os.system('./serve_file.sh')
        print('server is up')
        while not os.path.exists('pid'):
            pass
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
        print(f"{_url} will be played")
        self.furhat.say(url=_url,lipsync=True)

    def stop(self)->None:
        """Stops the robot from playing music
        """
        self.furhat.say_stop()

    def _get_file_path(self,music_item:str)->str:
        """Generates the url for a file from `self.music_list`

        Args:
            music_item (str): item of `self.music_list`

        Returns:
            str: url
        """
        return f"{self.src_path}/{music_item}"

    def _get_url(self)->str:
        """Returns url of the current indexed song

        Returns:
            str: url
        """
        return self._get_file_path(self.music_list[self.index])
    
    def next(self,music_playing=False)->None:
        """Handle gesture C
        If a music is currently playing, we start play the following
        Otherwise we only increase the index

        Args:
            music_playing (bool, optional): Wheter or not a music is currently played. Defaults to False.
        """
        self._increase_index()
        if music_playing:
            self.stop()
            self.play()
        print(f"current song : {self.music_list[self.index]}")

    def previous(self,music_playing=False)->None:
        """Handle gesture L
        If a music is currently playing, we start play the previous
        Otherwise we only decrease the index

        Args:
            music_playing (bool, optional): [description]. Defaults to False.
        """
        self._decrease_index()
        if music_playing:
            self.stop()
            self.play()
        print(f"current song : {self.music_list[self.index]}")
            


    

    def __del__(self):
        os.system(f'kill {self.pid}')
        print('server has been killed')

        

if __name__ == "__main__":
    player = AudioPlayer()

