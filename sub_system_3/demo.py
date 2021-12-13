from AudioPlayer import AudioPlayer
from time import sleep


def main():
    # Instantiate the audio player
    player = AudioPlayer()
    sleep(5)

    # Play the first music for 10s
    player.play()
    player.add_favorite()
    sleep(15)

    # Go forward 3 times and add 2nd song to favorite
    player.next()
    sleep(10)

    player.stop()
    player.next()
    sleep(5)

    player.play()
    sleep(10)
    player.previous()
    sleep(10)

    # Play favorite
    player.play_favorite()
    sleep(15)

    # Stop
    player.stop()

    player.exit()


if __name__ == "__main__":
    main()
