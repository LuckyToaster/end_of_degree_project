#!/bin/bash
#pandoc "$1" -o "${1%.*}.pdf" --template eisvogel --listings --pdf-engine=lualatex --toc --number-sections --resource-path=.:./images

#--variable classoption=twocolumn \

pandoc "$1" -o "${1%.*}.pdf" \
    --pdf-engine=lualatex \
    --number-sections \
    --variable papersize=a4 \
    --citeproc --bibliography=refs.bib \
    --variable documentclass=scrartcl
