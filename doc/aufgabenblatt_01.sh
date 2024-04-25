#!/bin/bash
pandoc --template ./eisvogel.tex aufgabenblatt_01.md --number-sections --listings --pdf-engine=lualatex  -o ./pdfs/Aufgabenblatt_01.pdf 
