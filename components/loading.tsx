import { Card } from "@/components/ui/card";
import HashLoader from "react-spinners/HashLoader";

const LoadingPage = () => {
  return (
    <div className="fixed top-0 left-0 right-0 bottom-0 backdrop-blur-sm backdrop-brightness-[30%] z-50 flex items-center justify-center">
      <Card className="p-12 px-16 flex flex-col justify-center items-center w-fit">
        <HashLoader color="#6D58C6" size={120} />
        <p className="text-[#23314C] font-bold text-3xl mt-8 text-center">
          Thinking...
        </p>
      </Card>
    </div>
  );
};

export default LoadingPage;
