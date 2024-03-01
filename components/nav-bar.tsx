"use client";

import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuContent,
  NavigationMenuTrigger,
  NavigationMenuList,
  navigationMenuTriggerStyle,
} from "./ui/navigation-menu";

import { FaGithub } from "react-icons/fa";
import Link from "next/link";
import { Button } from "./ui/button";

const NavbarPage = () => {
  return (
    <nav className="navbar container pt-8">
      <a className="flex-1" href="/">
        <img
          src="/logo/molar-logo-2.svg"
          alt="molar-logo-2"
          className="size-12"
        />
        <div className="btn btn-ghost text-2xl font-bold">Molar Support</div>
      </a>
      <div className="flex-none">
        <ul className="menu menu-horizontal px-1 items-center gap-x-8">
          <li>
            <a className="text-xl font-semibold">About</a>
          </li>
          <li>
            <a className="text-xl font-semibold" href="/history">History</a>
          </li>
          <li>
            <a
              href="https://github.com/dr-dolomite/MolarSupport.git"
              target="_blank"
              className="p-2 border-[#23314C] border-2 rounded-xl"
            >
              <FaGithub className="size-6" />
            </a>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default NavbarPage;