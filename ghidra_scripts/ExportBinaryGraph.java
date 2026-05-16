import ghidra.app.script.GhidraScript;
import ghidra.program.model.pcode.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.address.Address;
import ghidra.program.model.block.*;
import ghidra.program.model.symbol.*;
import ghidra.app.decompiler.*;

import java.util.*;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.File;

public class ExportBinaryGraph extends GhidraScript {
    @Override
    public void run() throws Exception {
        DecompInterface decomplib = new DecompInterface();
        decomplib.openProgram(currentProgram);

        String binaryName = currentProgram.getName();
        String outputPath = System.getProperty("user.home") + File.separator + binaryName + "_asm_graph.json";
        PrintWriter out = new PrintWriter(new FileWriter(outputPath));

        out.println("{"); 
        out.println("  \"functions\": ["); 
        
        Function func = getFirstFunction();
        boolean firstFunction = true;
        
        Listing listing = currentProgram.getListing();
        BasicBlockModel blockModel = new BasicBlockModel(currentProgram);

        Set<String> globalCallEdges = new LinkedHashSet<>();

        while (func != null && !monitor.isCancelled()) {
            if (func.isExternal() || func.isThunk()) {
                func = getFunctionAfter(func);
                continue;
            }

            CodeBlockIterator bbIter = blockModel.getCodeBlocksContaining(func.getBody(), monitor);
            
            if (bbIter.hasNext()) {
                if (!firstFunction) out.println(",");
                
                Set<Function> calledFunctions = func.getCalledFunctions(monitor);
                for (Function calledFunc : calledFunctions) {
                    String srcName = func.getName();
                    String tgtName = calledFunc.getName();
                    globalCallEdges.add(String.format("    {\"source\": \"%s\", \"target\": \"%s\"}", escapeJson(srcName), escapeJson(tgtName)));
                }

                List<String> nodesJson = new ArrayList<>();
                Set<String> edgesJson = new LinkedHashSet<>(); 
                
                Map<CodeBlock, Integer> blockToId = new HashMap<>();
                Map<Address, Integer> addrToBlockId = new HashMap<>();
                List<CodeBlock> nativeBlocks = new ArrayList<>();
                int blockIdCounter = 0;

                // --- 步骤 1：提取原生基本块，获取地址上下文以适配 CLAP ---
                while (bbIter.hasNext()) {
                    CodeBlock block = bbIter.next();
                    int currentBlockId = blockIdCounter++;
                    blockToId.put(block, currentBlockId);
                    nativeBlocks.add(block);
                    
                    List<String> asmList = new ArrayList<>();
                    InstructionIterator instIter = listing.getInstructions(block, true);
                    while (instIter.hasNext()) {
                        Instruction inst = instIter.next();
                        Address instAddr = inst.getMinAddress();
                        addrToBlockId.put(instAddr, currentBlockId);
                        
                        String asmStr = formatInstruction(inst, currentProgram);
                        asmStr = escapeJson(asmStr); // 使用专门的转义函数
                        
                        // [新增核心逻辑]: 提取控制流的跳转目标地址 (用于 CLAP 替换 INSTR<N>)
                        String targetAddrStr = "null";
                        Address[] flows = inst.getFlows();
                        if (flows != null && flows.length > 0) {
                            for (Address flowAddr : flows) {
                                // 排除顺延执行 (FallThrough) 的地址，只保留真正的跳转/调用目标
                                if (!flowAddr.equals(inst.getFallThrough())) {
                                    targetAddrStr = "\"" + flowAddr.toString() + "\"";
                                    break;
                                }
                            }
                        }
                        
                        // 组装成对象格式: {"addr": "...", "asm": "...", "target_addr": ...}
                        String instJson = String.format("{\"addr\": \"%s\", \"asm\": \"%s\", \"target_addr\": %s}", 
                                                        instAddr.toString(), asmStr, targetAddrStr);
                        asmList.add(instJson);
                    }
                    
                    String instructionsArray = "[\n            " + String.join(",\n            ", asmList) + "\n          ]";
                    nodesJson.add(String.format("{\"id\": %d, \"instructions\": %s}", currentBlockId, instructionsArray));
                }

                // --- 步骤 2：提取原生 CFG ---
                for (CodeBlock block : nativeBlocks) {
                    int srcId = blockToId.get(block);
                    CodeBlockReferenceIterator destIter = block.getDestinations(monitor);
                    while (destIter.hasNext()) {
                        CodeBlock destBlock = destIter.next().getDestinationBlock();
                        if (blockToId.containsKey(destBlock)) {
                            int destId = blockToId.get(destBlock);
                            edgesJson.add(String.format("{\"source\": %d, \"target\": %d, \"relation\": 0}", srcId, destId));
                        }
                    }
                }

                // --- 步骤 3：提取跨块 DFG ---
                DecompileResults res = decomplib.decompileFunction(func, 60, monitor);
                HighFunction highFunc = res.getHighFunction();
                if (highFunc != null) {
                    Iterator<PcodeOpAST> ops = highFunc.getPcodeOps();
                    while (ops.hasNext()) {
                        PcodeOpAST op = ops.next();
                        Address tgtAddr = op.getSeqnum().getTarget();
                        
                        if (tgtAddr != null && tgtAddr.isMemoryAddress()) {
                            Integer tgtBlockId = addrToBlockId.get(tgtAddr);
                            if (tgtBlockId != null) {
                                Varnode[] inputs = op.getInputs();
                                for (Varnode input : inputs) {
                                    if (input.getDef() != null) {
                                        Address srcAddr = input.getDef().getSeqnum().getTarget();
                                        if (srcAddr != null && srcAddr.isMemoryAddress()) {
                                            Integer srcBlockId = addrToBlockId.get(srcAddr);
                                            if (srcBlockId != null && !srcBlockId.equals(tgtBlockId)) {
                                                edgesJson.add(String.format("{\"source\": %d, \"target\": %d, \"relation\": 1}", srcBlockId, tgtBlockId));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // --- 步骤 4：写入单个函数的内部图 JSON ---
                out.println("    {");
                out.println("      \"function\": \"" + escapeJson(func.getName()) + "\",");
                out.println("      \"nodes\": [");
                for (int i = 0; i < nodesJson.size(); i++) {
                    out.print("        " + nodesJson.get(i) + (i < nodesJson.size() - 1 ? ",\n" : "\n"));
                }
                out.println("      ],");
                out.println("      \"edges\": [");
                int edgeCount = 0;
                for (String edgeStr : edgesJson) {
                    out.print("        " + edgeStr + (edgeCount < edgesJson.size() - 1 ? ",\n" : "\n"));
                    edgeCount++;
                }
                out.println("      ]");
                out.print("    }");
                firstFunction = false;
            }
            func = getFunctionAfter(func);
        }
        
        out.println("\n  ],"); // 闭合 functions 数组
        
        out.println("  \"call_graph\": [");
        int cgCount = 0;
        for (String cgEdge : globalCallEdges) {
            out.print(cgEdge + (cgCount < globalCallEdges.size() - 1 ? ",\n" : "\n"));
            cgCount++;
        }
        out.println("  ]");
        
        out.println("}"); // 闭合最外层的大对象
        
        out.close();
        decomplib.dispose();
        
        System.out.println(">>> 成功导出高精度图数据 (含跳转地址锚点) 至: " + outputPath);
    }

    private String escapeJson(String str) {
        if (str == null) return "";
        return str.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private String formatInstruction(Instruction inst, Program program) {
        StringBuilder sb = new StringBuilder();
        sb.append(inst.getMnemonicString());
        
        int numOperands = inst.getNumOperands();
        SymbolTable symTable = program.getSymbolTable();
        
        for (int i = 0; i < numOperands; i++) {
            sb.append(i == 0 ? " " : ", ");
            String opRep = inst.getDefaultOperandRepresentation(i);
            
            Reference[] refs = inst.getOperandReferences(i);
            if (refs != null && refs.length > 0) {
                for (Reference ref : refs) {
                    Address refAddr = ref.getToAddress();
                    if (refAddr.isMemoryAddress()) {
                        Symbol sym = symTable.getPrimarySymbol(refAddr);
                        if (sym != null && !sym.isDynamic() && !sym.getName().startsWith("LAB_") && !sym.getName().startsWith("DAT_")) {
                            if (inst.getFlowType().isCall() || inst.getFlowType().isJump() || opRep.startsWith("0x")) {
                                opRep = sym.getName();
                                break;
                            }
                        }
                    }
                }
            }
            sb.append(opRep);
        }
        return sb.toString();
    }
}