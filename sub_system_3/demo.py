from AudioPlayer import AudioPlayer
from time import sleep


def main():
    player = AudioPlayer()

    player.start()
    sleep(3)

    player.play()
    sleep(5)
    player.add_favorite()
    sleep(5)

    player.next()
    sleep(7)

    player.stop()
    sleep(2)
    player.next()
    sleep(1)
    player.next()
    sleep(1)

    player.play()
    sleep(6)
    player.previous()
    sleep(6)

    player.play_favorite()
    sleep(5)

    player.stop()

    player.exit()

    player = AudioPlayer()
    player.play()
    player.add_favorite()
    player.next()
    player.stop()
    player.next()
    player.next()
    player.play()
    player.previous()
    player.play_favorite()
    player.stop()


if __name__ == "__main__":
    main()



    
