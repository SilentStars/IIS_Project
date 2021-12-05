from AudioPlayer import AudioPlayer
from time import sleep

def main():
    # Instantiate the audio player
    player = AudioPlayer()

    # Play the first music for 10s
    player.play()
    sleep(20)

    # Stop the music
    player.stop()
    sleep(3)

    # Go forward 3 times and add 2nd song to favorite
    player.next()
    player.add_favorite()
    player.next()
    player.next()

    # Play for 10 seconds
    player.play()
    sleep(20)

    # Go backward of 1 song
    player.previous()
    sleep(20)

    # Play favorite
    player.play_favorite()
    sleep(20)

    # Stop
    player.stop()



if __name__ == "__main__":
    main()