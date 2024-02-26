import { LoginButton } from "@/components/auth/login-button";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center space-y-8">
      <h1 className="text-5xl font-semibold text-wrap"> Welcome to Molar Support </h1>
      <LoginButton>
      <Button variant="default" size="lg">
        Get Started
      </Button>
      </LoginButton>
     
    </main>
  );
}
