from manim import config
config.background_color = "WHITE"

from manim import Scene, MathTex, Indicate, RED, BLACK

class ModularEquationHighlight(Scene):
    def construct(self):
        tokens = [
            "(", "(", "(", "(", "(", "(", "(",  # 0-6
            "5", "-", "6",                      # 7-9
            ")", ")",                           # 10-11
            "\mod", "10",                      # 12-13
            ")", "+", "5",                      # 14-16
            ")", ")", "\mod", "10",            # 17-20
            ")", "+", "5",                      # 21-23
            ")", "\mod", "10", "=", "9"        # 24-28
        ]

        eq = MathTex(*tokens)
        eq.set_color(BLACK)
        eq.scale(1.1)

        self.add(eq)
        self.wait(40 / 15)

        highlight_sequence = [7, 8, 9, 15, 16, 22, 23]  # 9 + 9, +1, +3

        for idx in highlight_sequence:
            eq[idx].set_stroke(color=RED, width=6)
            self.play(Indicate(eq[idx], color=RED, scale_factor=1.3), run_time=10 / 15)
            eq[idx].set_stroke(color=BLACK, width=0)

        self.wait(10 / 15)
