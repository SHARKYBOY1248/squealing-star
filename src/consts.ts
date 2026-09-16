import type { Metadata, Site, Socials } from "@types";

export const SITE: Site = {
  TITLE: "Ajit Kadaveru",
  DESCRIPTION: "Software engineer in San Francisco, CA.",
  EMAIL: "ajit.kadaveru@berkeley.edu",
  NUM_POSTS_ON_HOMEPAGE: 3,
};

export const HOME: Metadata = {
  TITLE: "Home",
  DESCRIPTION: "Software engineer in San Francisco, CA.",
};

export const BLOG: Metadata = {
  TITLE: "Blog",
  DESCRIPTION: "Notes on ML systems, inference, and things I'm learning.",
};

export const SOCIALS: Socials = [
  {
    NAME: "X",
    HREF: "https://x.com/ajit_k02",
  },
  {
    NAME: "LinkedIn",
    HREF: "https://www.linkedin.com/in/ajit-kadaveru-a11125180/",
  },
  {
    NAME: "YouTube",
    HREF: "https://www.youtube.com/@sharkyboy1248/",
  },
  {
    NAME: "Codeforces",
    HREF: "https://codeforces.com/profile/2020akadaver",
  },
];
