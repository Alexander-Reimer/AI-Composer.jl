# AI at music - Computers learn composing
This is the result of a project by MatteoFriedrich (Matteo Friedrich, 13, Gymnasium Eversten Oldenburg) and me (Alexander Reimer, 14, Gymnasium Eversten Oldenburg) for the Jugend Forscht 2021 competition in Germany which is called "Künstliche Intelligenz in der Musik - Computer lernen Komponieren". It is capable of creating new, mostly original Jazz music by using an Auto Encoder. If you want to know more about it, you can read "paper (German).pdf" but because we switched to using an Autoencoder instead of Supervised Learning after the deadline for it the currently used approach is only described roughly in a single paragraph in "Diskussion".

If you have any questions, feel free to email me (alexander.reimer2357@gmail.com). If you have any problems or encounter a bug, please create a new issue.

## Installation
Windows is recommended. Linux should work just as well but hasn't been tested yet. MacOS was tested with an older version and should work with the newest one as well, but it has performance issues because the package we are using for the UI (Gtk) isn't very compatible with it.

You can find all releases [here](https://github.com/AR102/AI-Composer.jl/releases). We recommend using the newest version (v1.2-pre1).
[Install it](https://github.com/AR102/AI-Composer.jl/releases/tag/v1.2-pre1), move it to a preffered location and unzip it.

### Windows
Open the folder "Neural Jazz" and execute the Batch file ("Neural Jazz.bat"). The first startup may take a while as a few packages have to be installed. An internet connection is necessary but only for the first time.

### MacOS and Linux
[Install the Julia programming language](https://julialang.org/downloads/oldreleases/) (currently only version 1.5.4 is tested). Then start it by typing `julia` into your systems terminal. Now navigate to the location of the folder by typing `cd("path/to/Neural Jazz/src")`. To start the program, type `include("main.jl")`. The first startup may take a while as a few packages have to be installed. An internet connection is necessary but only for the first time.

## Usage
Change the features of the neuronal network by changing the values of the five sliders in the top right. You can set them to random values with the "Random Seed" Button below.

The output of the neuronal is shown in the top left as a 16 high and 21 wide Array / "grid". The width is the position of a note (an eighth note each) and the height the tone of a note (from C-4 at the bottom to C-6 at the top). If the rectangle at a specific position is white, it means that there is no note played there. If it's black, it is played. The shade of black / grey is determined by how probable the neuronal network thinks it is that the note is "right" (sounds good). It ranges from 0.0 (definitely not - white) to 1.0 (definitely - black) with 0.5 (maybe - grey) in the middle.

The slider in the middle left is the "tendency" slider. It determines what notes are going to be shown in the bottom left: The "probability" of a note has to be higher than the tendency for it to be shown there.

You can hear the notes there by clicking "Play" and pause playback by clicking "Pause" or use the "P" key. The "Reset" button just pauses the music and starts the music from the beginning. The slider shows the playing progress. You can move it to skip forward or backward. 

The sound quality isn't very good. To hear the song in higher quality with a piano as an instrument, you have to choose different motives or "modules". You can save the current notes to module x by clicking the "Keep as Module x" button and delete the notes of module x by clicking the "Reset Module x" button. Once you have all four modules set, you can click "Save Song (all modules)". It'll composite the four modules into the pattern 1, 1, 2, 1, 3, 4, 2, 1, 4, 1 as each module is only 2 tacts long. After you chose the location, the midi file of the song will be saved there. If you want to change the tempo of it, look at the notes or export it as a different file, you can use another program like [musescore](https://musescore.org/de) to open and edit the file.

## Last updated: 04-06-2021 (v1.2-pre1)
