import numpy as np

shot_cool_down = 5
diver_move_interval = 5
enemy_shot_interval = 10
max_oxygen = 200
# Update environment according to agent action
def step_agent_numpy(env, action):
    a = env.env.action_map[action]

    # Resolve player action
    if(a=='f' and env.env.shot_timer == 0):
        env.env.f_bullets+=[[env.env.sub_x, env.env.sub_y, env.env.sub_or]]
        env.env.shot_timer = shot_cool_down
    elif(a=='l'):
        env.env.sub_x = max(0, env.env.sub_x-1)
        env.env.sub_or = False
    elif(a=='r'):
        env.env.sub_x = min(9, env.env.sub_x+1)
        env.env.sub_or = True
    elif(a=='u'):
        env.env.sub_y = max(0, env.env.sub_y-1)
    elif(a=='d'):
        env.env.sub_y = min(8, env.env.sub_y+1)


def step_bullets_numpy(env):
    # Update friendly Bullets
    r = 0
    for bullet in reversed(env.env.f_bullets):
        bullet[0]+=1 if bullet[2] else -1
        if(bullet[0]<0 or bullet[0]>9):
            env.env.f_bullets.remove(bullet)
        else:
            removed = False
            for x in env.env.e_fish:
                if(bullet[0:2]==x[0:2]):
                    env.env.e_fish.remove(x)
                    env.env.f_bullets.remove(bullet)
                    r+=1
                    removed = True
                    break
            if(not removed):
                for x in env.env.e_subs:
                    if(bullet[0:2]==x[0:2]):
                        env.env.e_subs.remove(x)
                        env.env.f_bullets.remove(bullet)
                        r+=1
    return r


def step_divers_numpy(env):
    # Update divers
    for diver in reversed(env.env.divers):
        if(diver[0:2]==[env.env.sub_x,env.env.sub_y]
           and env.env.diver_count<6):
            env.env.divers.remove(diver)
            env.env.diver_count+=1
        else:
            if(diver[3]==0):
                diver[3]=diver_move_interval
                diver[0]+=1 if diver[2] else -1
                if(diver[0]<0 or diver[0]>9):
                    env.env.divers.remove(diver)
                elif(diver[0:2]==[env.env.sub_x,env.env.sub_y]
                     and env.env.diver_count<6):
                    env.env.divers.remove(diver)
                    env.env.diver_count+=1
            else:
                diver[3]-=1


def step_e_subs_numpy(env, r):
    # Update enemy subs
    for sub in reversed(env.env.e_subs):
        if(sub[0:2]==[env.env.sub_x,env.env.sub_y]):
            env.env.terminal = True
        if(sub[3]==0):
            sub[3]=env.env.move_speed
            sub[0]+=1 if sub[2] else -1
            if(sub[0]<0 or sub[0]>9):
                env.env.e_subs.remove(sub)
            elif(sub[0:2]==[env.env.sub_x,env.env.sub_y]):
                env.env.terminal = True
            else:
                for x in env.env.f_bullets:
                    if(sub[0:2]==x[0:2]):
                        env.env.e_subs.remove(sub)
                        env.env.f_bullets.remove(x)
                        r+=1
                        break
        else:
            sub[3]-=1
        if(sub[4]==0):
            sub[4]=enemy_shot_interval
            env.env.e_bullets+=[[sub[0] if sub[2] else sub[0], sub[1], sub[2]]]
        else:
            sub[4]-=1
    return r


def step_e_bullets_numpy(env, r):
    # Update enemy bullets
    for bullet in reversed(env.env.e_bullets):
        if(bullet[0:2]==[env.env.sub_x,env.env.sub_y]):
            env.env.terminal = True
        bullet[0]+=1 if bullet[2] else -1
        if(bullet[0]<0 or bullet[0]>9):
            env.env.e_bullets.remove(bullet)
        else:
            if(bullet[0:2]==[env.env.sub_x,env.env.sub_y]):
                env.env.terminal = True

    # Update enemy fish
    for fish in reversed(env.env.e_fish):
        if(fish[0:2]==[env.env.sub_x,env.env.sub_y]):
            env.env.terminal = True
        if(fish[3]==0):
            fish[3]=env.env.move_speed
            fish[0]+=1 if fish[2] else -1
            if(fish[0]<0 or fish[0]>9):
                env.env.e_fish.remove(fish)
            elif(fish[0:2]==[env.env.sub_x,env.env.sub_y]):
                env.env.terminal = True
            else:
                for x in env.env.f_bullets:
                    if(fish[0:2]==x[0:2]):
                        env.env.e_fish.remove(fish)
                        env.env.f_bullets.remove(x)
                        r+=1
                        break
        else:
            fish[3]-=1
    return r


def step_timers_numpy(env, r):
    # Update various timers
    env.env.e_spawn_timer -= env.env.e_spawn_timer>0
    env.env.d_spawn_timer -= env.env.d_spawn_timer>0
    env.env.shot_timer -= env.env.shot_timer>0
    if(env.env.oxygen<0):
        env.env.terminal = True
    if(env.env.sub_y>0):
        env.env.oxygen-=1
        env.env.surface = False
    else:
        if(not env.env.surface):
            if(env.env.diver_count == 0):
                env.env.terminal = True
            else:
                r+=env.env._surface()
    return r, env.env.terminal


def surface_numpy(env):
    env.env.surface = True
    if(env.env.diver_count == 6):
        env.env.diver_count = 0
        r = env.env.oxygen*10//max_oxygen
    else:
        r = 0
    env.env.oxygen = max_oxygen
    env.env.diver_count -= 1
    if env.env.ramping and (env.env.e_spawn_speed>1 or env.env.move_speed>2):
        if(env.env.move_speed>2 and env.env.ramp_index%2):
                env.env.move_speed-=1
        if(env.env.e_spawn_speed>1):
                env.env.e_spawn_speed-=1
        env.env.ramp_index+=1
    return r
