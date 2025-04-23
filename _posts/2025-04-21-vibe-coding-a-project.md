---
layout: post
author: michael
title:  "Vibe Coding a Project in 2025"
date:   2025-04-21 06:00:00 +0530
categories: VibeCoding Cursor PersonalProject
---

This is a post of a bit of a different flavor. I'm going to implement a project from scratch and document my learnings here. Recently the term "vibe coding" was invented by Andrey Karphaty, this is my attempt at doing a personal project in that fasion.

I have not developed much outside of Googles ecosystem for over three years, so this will include learning a lot of new stuff. Moreover I have only ever worked as so called backend developer, more specifically as machine learning engineer. So I'm completely unfamiliar with what I imagine this project to include:

* Frontend website
* Backend uploading and data processing function
* Authentication
* Payment options
* ML outside of either local projects or Google internal infrastructure


What will I build: A website that takes an album of images, removes low quality images and duplicates, selects the best pictures automatically. No more manual sifiting through your vacation pictures!

# Part One - Local App

Asking Gemini 2.5 Pro about how do do this at all.

# Cursor Agent

I have no idea how to do a web application, I just created the frontend and backend folder. Cursor Agent is very helpful in filling it with content. But a first test already fails at the login, there is no hardcoded login, I have to ask for that myself (probably for good reason). It takes a few turns with the agent to solve this, database user was not created and the backend could not communicate with the frontend.

Once this is solved the progress is rapid, I can ask for new features and they are implemented with ease.

I have too much stuff after compiling for the first time, cursor throws an error in the view that shows the git status. Luckily cursor can also generate a decent .gitignore file, but even with this change it still refuses to show me the git view.





