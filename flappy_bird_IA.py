import pygame
import neat
import time
import os
import random
import matplotlib.pyplot as plt
pygame.font.init()

janela_x = 500
janela_y = 800

current_path = os.path.dirname(__file__)
image_path = os.path.join(current_path, 'imgs')

sprites_passaro = [pygame.transform.scale2x(pygame.image.load(os.path.join(image_path, 'bird1.png'))), pygame.transform.scale2x(
    pygame.image.load(os.path.join(image_path, 'bird2.png'))), pygame.transform.scale2x(pygame.image.load(os.path.join(image_path, 'bird3.png')))]
sprite_cano = pygame.transform.scale2x(
    pygame.image.load(os.path.join(image_path, 'pipe.png')))
sprite_fundo = pygame.transform.scale2x(
    pygame.image.load(os.path.join(image_path, 'bg.png')))
sprite_chao = pygame.transform.scale2x(
    pygame.image.load(os.path.join(image_path, 'base.png')))
STAT_FONT = pygame.font.SysFont('comicsans', 50)


class Passaro:
    imgs = sprites_passaro
    rot_max = 25
    rot_vel = 20
    tempo_ani = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.altura = self.y
        self.img_frame = 0
        self.img = self.imgs[0]

    def pulo(self):
        self.vel = -5.5
        self.tick_count = 0
        self.altura = self.y

    def movimento(self):
        self.tick_count += 1

        d = self.vel*self.tick_count + 1.5*self.tick_count**2

        if d >= 16:
            d = 16

        if d < 0:
            d -= 2

        self.y = self.y + d
        if d < 0 or self.y < self.altura + 50:
            if self.tilt < self.rot_max:
                self.tilt = self.rot_max
        else:
            if self.tilt > -90:
                self.tilt -= self.rot_vel

    def draw(self, win):
        self.img_frame += 1

        if self.img_frame < self.tempo_ani:
            self.img = self.imgs[0]
        elif self.img_frame < self.tempo_ani*2:
            self.img = self.imgs[1]
        elif self.img_frame < self.tempo_ani*3:
            self.img = self.imgs[2]
        elif self.img_frame < self.tempo_ani*4:
            self.img = self.imgs[1]
        elif self.img_frame < self.tempo_ani*4+1:
            self.img = self.imgs[0]
            self.img_frame = 0

        if self.tilt <= -80:
            self.img = self.imgs[1]
            self.img_frame = self.tempo_ani*2

        rotate = pygame.transform.rotate(self.img, self.tilt)
        reta = rotate.get_rect(center=self.img.get_rect(
            topleft=(self.x, self.y)).center)
        win.blit(rotate, reta.topleft)

    def pixel_mask(self):
        return pygame.mask.from_surface(self.img)


class Cano:
    ESPACO = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.altura = 0
        self.espaco = 100

        self.topo = 0
        self.base = 0
        self.TOPO_CANO = pygame.transform.flip(sprite_cano, False, True)
        self.BASE_CANO = sprite_cano

        self.passou = False
        self.def_altura()

    def def_altura(self):
        self.altura = random.randrange(50, 450)
        self.topo = self.altura - self.TOPO_CANO.get_height()  # Talvez mude
        self.base = self.altura + self.ESPACO

    def movimento(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.TOPO_CANO, (self.x, self.topo))
        win.blit(self.BASE_CANO, (self.x, self.base))

    def colisao(self, passaro):
        passaro_hitbox = passaro.pixel_mask()
        topo_hitbox = pygame.mask.from_surface(self.TOPO_CANO)
        c_base_hitbox = pygame.mask.from_surface(self.BASE_CANO)

        topo_offset = (self.x - passaro.x, self.topo - round(passaro.y))
        base_offset = (self.x - passaro.x, self.base - round(passaro.y))

        b_point = passaro_hitbox.overlap(c_base_hitbox, base_offset)
        t_point = passaro_hitbox.overlap(topo_hitbox, topo_offset)

        if t_point or b_point:
            return True

        return False


class Chao:
    VEL = 5
    LARG = sprite_chao.get_width()
    IMG = sprite_chao

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.LARG

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.LARG < 0:
            self.x1 = self.x2 + self.LARG

        if self.x2 + self.LARG < 0:
            self.x2 = self.x1 + self.LARG

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, passaros, canos, chao, score):
    win.blit(sprite_fundo, (0, 0))

    for cano in canos:
        cano.draw(win)

    texto = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(texto, (janela_x - 10 - texto.get_width(), 10))
    draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg')

    chao.draw(win)
    for passaro in passaros:
        passaro.draw(win)

    pygame.display.update()


def main(genomes, config):
    nets = []
    ge = []
    passaros = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        passaros.append(Passaro(230, 350))
        g.fitness = 0
        ge.append(g)

    chao = Chao(730)
    canos = [Cano(600)]
    win = pygame.display.set_mode((janela_x, janela_y))
    clock = pygame.time.Clock()
    score = 0

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        cano_ind = 0

        if len(passaros) > 0:
            if len(canos) > 1 and passaros[0].x > canos[0].x + canos[0].TOPO_CANO.get_width():
                cano_ind = 1

        else:
            run = False
            break

        for x, passaro in enumerate(passaros):
            passaro.movimento()
            ge[x].fitness += 0.1

            output = nets[x].activate((passaro.y, abs(
                passaro.y - canos[cano_ind].altura), abs(passaro.y - canos[cano_ind].base)))

            if output[0] > 0.5:
                passaro.pulo()

        rem = []

        add_cano = False

        for cano in canos:
            for x, passaro in enumerate(passaros):
                if cano.colisao(passaro):
                    ge[x].fitness -= 1
                    passaros.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not cano.passou and cano.x < passaro.x:
                    cano.passou = True
                    add_cano = True

            if cano.x + cano.TOPO_CANO.get_width() < 0:
                rem.append(cano)

            cano.movimento()

        if add_cano:
            score += 1
            for g in ge:
                g.fitness += 5
            canos.append(Cano(700))

        for r in rem:
            canos.remove(r)

        for x, passaro in enumerate(passaros):
            if passaro.y + passaro.img.get_height() >= 730 or passaro.y < 0:
                passaros.pop(x)
                nets.pop(x)
                ge.pop(x)

        chao.move()
        draw_window(win, passaros, canos, chao, score)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    vencedor = pop.run(main, 100)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = (os.path.join(local_dir, "config-feedforward.txt"))
    run(config_path)
