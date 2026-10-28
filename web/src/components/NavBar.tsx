"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Classify" },
  { href: "/history", label: "History" },
  { href: "/stats", label: "Stats" },
];

export default function NavBar() {
  const pathname = usePathname();

  return (
    <header className="border-b border-zinc-200 bg-white">
      <nav className="mx-auto flex max-w-5xl items-center gap-8 px-6 py-4">
        <Link href="/" className="text-lg font-semibold tracking-tight text-emerald-700">
          TerraSense
        </Link>
        <ul className="flex gap-5 text-sm">
          {LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <li key={link.href}>
                <Link
                  href={link.href}
                  className={active ? "font-medium text-zinc-900" : "text-zinc-500 hover:text-zinc-900"}
                >
                  {link.label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}
