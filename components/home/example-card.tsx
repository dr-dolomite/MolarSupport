import {
  Card,
  CardHeader,
  CardFooter,
  CardContent,
  CardDescription,
  CardTitle,
} from "@/components/ui/card";

import Link from "next/link";

const ExampleCard = () => {
  return (
    <Card className="px-6 py-2 shadow-md drop-shadow-md flex flex-col justify-center">
      <CardHeader>
        <CardTitle>
          <p className="text-[#878787] text-xl font-normal">
            You can try these sample cases:
          </p>
        </CardTitle>
        <CardContent className="flex flex-row items-center justify-center py-2 gap-x-6">
          <Link href="/">
            <img
              src="/samples/sample1.png"
              alt="sample-1"
              className="size-32 object-cover"
            />
          </Link>

          <Link href="/">
            <img
              src="/samples/sample2.png"
              alt="sample-2"
              className="size-32 object-cover"
            />
          </Link>

          <Link href="/">
            <img
              src="/samples/sample3.png"
              alt="sample-3"
              className="size-32 object-cover"
            />
          </Link>

          <Link href="/">
            <img
              src="/samples/sample4.png"
              alt="sample-4"
              className="size-32 object-cover"
            />
          </Link>
        </CardContent>
      </CardHeader>
    </Card>
  );
};

export default ExampleCard;
