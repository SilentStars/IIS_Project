package furhatos.app.music_controller

import furhatos.app.music_controller.flow.*
import furhatos.skills.Skill
import furhatos.flow.kotlin.*

class Music_controllerSkill : Skill() {
    override fun start() {
        Flow().run(Idle)
    }
}

fun main(args: Array<String>) {
    Skill.main(args)
}
