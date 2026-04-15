"use client";
import { useState } from "react";
import { api } from "@/utils/api";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function Register() {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const router = useRouter();

    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            const res = await api.post("/auth/register", { username, password });
            if (res.message === "User registered") {
                alert("Registration successful! Please login.");
                router.push("/auth/login");
            } else {
                alert("Registration failed: " + res.message);
            }
        } catch (err) {
            alert("Registration error");
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-neutral-950 text-white">
            <div className="bg-neutral-900 p-8 rounded-2xl shadow-xl w-full max-w-md border border-neutral-800">
                <h2 className="text-3xl font-bold mb-6 text-center">Sign Up</h2>
                <form onSubmit={handleRegister} className="flex flex-col gap-4">
                    <input
                        type="text"
                        placeholder="Username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        className="p-3 rounded bg-neutral-800 border border-neutral-700 focus:border-blue-500 outline-none transition-colors"
                    />
                    <input
                        type="password"
                        placeholder="Password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="p-3 rounded bg-neutral-800 border border-neutral-700 focus:border-blue-500 outline-none transition-colors"
                    />
                    <button type="submit" className="bg-green-600 hover:bg-green-700 text-white p-3 rounded font-bold transition-colors">
                        Register
                    </button>
                </form>
                <p className="mt-4 text-center text-neutral-400">
                    Already have an account? <Link href="/auth/login" className="text-blue-400 hover:underline">Log in</Link>
                </p>
            </div>
        </div>
    );
}