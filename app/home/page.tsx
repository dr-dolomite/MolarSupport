import AboutCard from "@/components/home/about-card";
import ExampleCard from "@/components/home/example-card";
import M3InputCard from "@/components/home/m3-input";
import McInputCard from "@/components/home/mc-input";
import React from "react";

const HomePage = () => {
  return (
    <div className="flex justfy-center items-center py-16 px-24">
        <div className="flex flex-row justify-center items-center space-x-16">
          <div className="w-[60%] flex flex-col gap-y-8">
            <AboutCard />
            <ExampleCard />
          </div>

          <div className="w-[40%] flex flex-col gap-y-8">
            <McInputCard />
            <M3InputCard />
          </div>
        </div>
    </div>
  );
};

export default HomePage;
